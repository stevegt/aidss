package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"

	"github.com/fsnotify/fsnotify"
	"github.com/google/uuid"
	"github.com/spf13/cobra"
	"rsc.io/pdf"

	"github.com/stevegt/aidss/llm"
	. "github.com/stevegt/goadapt"
)

var (
	mutex sync.Mutex // To handle concurrent access

	promptFn   = "prompt.txt"
	responseFn = "response.txt"
)

type Prompt struct {
	InFiles    []string
	OutFiles   []string
	SysMsg     string
	PromptText string
}

func main() {
	// Initialize and register providers
	llm.RegisterProviders()

	// Get available models from llm package
	models := llm.Models()

	// Initialize the command-line interface
	rootCmd := &cobra.Command{
		Use:   "decision_tool",
		Short: "Decision Support Tool",
		Run: func(cmd *cobra.Command, args []string) {
			watchPath, err := cmd.Flags().GetString("path")
			Ck(err)
			modelName, err := cmd.Flags().GetString("model")
			Ck(err)
			startDaemon(watchPath, modelName)
		},
	}

	// Make sure usage includes model names
	modelUsage := fmt.Sprintf("Model to use (%s)", strings.Join(models, ", "))

	// Define flags
	rootCmd.Flags().StringP("path", "p", ".", "Path to watch")
	rootCmd.Flags().StringP("model", "m", models[0], modelUsage)

	// Execute the root command
	if err := rootCmd.Execute(); err != nil {
		log.Fatal(err)
	}
}

// startDaemon starts the decision tool daemon. The daemon watches the file system for changes
// and responds to user messages and attachments.
func startDaemon(watchPath string, modelName string) {
	var err error

	// Set up the LLM client based on the model name
	client, err := llm.NewClient(modelName)
	if err != nil {
		log.Fatal(err)
	}

	// Start the file watcher
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		log.Fatal(err)
	}
	defer watcher.Close()

	done := make(chan bool)

	// Handle file system events
	go func() {
		for {
			select {
			case event, ok := <-watcher.Events:
				if !ok {
					// Watcher has been closed
					return
				}
				if event.Op&fsnotify.Write == fsnotify.Write {
					// handle file write events
					if filepath.Base(event.Name) == promptFn {
						log.Println("Detected change in:", event.Name)
						handleUserMessage(filepath.Dir(event.Name), client, watchPath)
					}
					if filepath.Ext(event.Name) == ".pdf" {
						log.Println("Detected PDF attachment:", event.Name)
						handlePDFAttachment(event.Name, extractTextFromPDF)
					}
				}
				if event.Op&fsnotify.Create == fsnotify.Create {
					// If a new directory is created, add it to the watcher
					fi, err := os.Stat(event.Name)
					if err == nil && fi.IsDir() {
						watcher.Add(event.Name)
						log.Println("Added new directory to watcher:", event.Name)
					}
				}
			case err, ok := <-watcher.Errors:
				if !ok {
					return
				}
				log.Println("error:", err)
			}
		}
	}()

	// Watch the root path
	err = addWatcherRecursive(watcher, watchPath)
	if err != nil {
		log.Fatal(err)
	}

	log.Println("Started watching:", watchPath)
	<-done
}

// addWatcherRecursive recursively adds a directory and its subdirectories to the watcher
func addWatcherRecursive(watcher *fsnotify.Watcher, path string) error {
	err := watcher.Add(path)
	if err != nil {
		return err
	}

	files, err := ioutil.ReadDir(path)
	if err != nil {
		return err
	}

	for _, file := range files {
		if file.IsDir() {
			err = addWatcherRecursive(watcher, filepath.Join(path, file.Name()))
			if err != nil {
				return err
			}
		}
	}

	return nil
}

// parsePromptFile parses the prompt file and returns a Prompt struct
func parsePromptFile(filename string) (*Prompt, error) {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	content := string(data)
	prompt := &Prompt{}

	// Split headers and prompt text
	parts := strings.SplitN(content, "\n\n", 2)
	if len(parts) != 2 {
		return nil, fmt.Errorf("Invalid prompt file format")
	}

	headers := parts[0]
	prompt.PromptText = parts[1]

	lines := strings.Split(headers, "\n")
	currentSection := ""
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		switch {
		case line == "In:":
			currentSection = "In"
		case line == "Out:":
			currentSection = "Out"
		case strings.HasPrefix(line, "Sysmsg:"):
			currentSection = "Sysmsg"
			prompt.SysMsg = strings.TrimSpace(strings.TrimPrefix(line, "Sysmsg:"))
		default:
			switch currentSection {
			case "In":
				prompt.InFiles = append(prompt.InFiles, line)
			case "Out":
				prompt.OutFiles = append(prompt.OutFiles, line)
			case "Sysmsg":
				prompt.SysMsg += "\n" + line
			default:
				// Ignore unknown sections
			}
		}
	}
	return prompt, nil
}

// handleUserMessage handles a user message by generating a response from the language model
func handleUserMessage(path string, client llm.Client, watchPath string) {
	mutex.Lock()
	defer mutex.Unlock()

	messagePath := filepath.Join(path, promptFn)
	prompt, err := parsePromptFile(messagePath)
	if err != nil {
		log.Println("Error parsing prompt file:", err)
		return
	}

	// Build context messages
	contextMessages := buildContextMessages(path, watchPath)

	// Add system message if provided
	if prompt.SysMsg != "" {
		contextMessages = append(contextMessages, llm.Message{
			Role:    llm.ChatMessageRoleSystem,
			Content: prompt.SysMsg,
		})
	}

	// Read and include contents of InFiles
	inFilesContent, err := readInFilesContent(prompt.InFiles, watchPath)
	if err != nil {
		log.Println("Error reading In files:", err)
		return
	}

	// Build the user message
	userContent := Spf("%s\n\n", prompt.PromptText)
	if len(inFilesContent) > 0 {
		userContent += "The following files are attached:\n" + inFilesContent + "\n"
	}

	// Append the new user message
	contextMessages = append(contextMessages, llm.Message{
		Role:    llm.ChatMessageRoleUser,
		Content: userContent,
	})

	response, err := getLLMResponse(contextMessages, client)
	if err != nil {
		log.Println("Error getting LLM response:", err)
		return
	}

	// Save the LLM response
	responsePath := filepath.Join(path, responseFn)
	err = ioutil.WriteFile(responsePath, []byte(response), 0644)
	if err != nil {
		log.Println("Error writing LLM response:", err)
	}

	log.Println("LLM response written to:", responsePath)

	// Parse the LLM response for updated files
	err = processLLMResponse(response, prompt.OutFiles, watchPath)
	if err != nil {
		log.Println("Error processing LLM response:", err)
	}
}

func readInFilesContent(inFiles []string, watchPath string) (string, error) {
	var contentBuilder strings.Builder
	parentDir := filepath.Dir(watchPath)
	for _, relPath := range inFiles {
		absPath := filepath.Join(parentDir, relPath)
		data, err := ioutil.ReadFile(absPath)
		if err != nil {
			return "", fmt.Errorf("error reading file %s: %v", absPath, err)
		}
		contentBuilder.WriteString(fmt.Sprintf("<IN filename=\"%s\">\n%s\n</IN>\n", relPath, string(data)))
	}
	return contentBuilder.String(), nil
}

func processLLMResponse(response string, outFiles []string, watchPath string) error {
	parentDir := filepath.Dir(watchPath)

	// Regular expression to match <OUT filename="...">...</OUT>
	re := regexp.MustCompile(`<OUT\s+filename="([^"]+)">\s*(?s)(.*?)</OUT>`)
	matches := re.FindAllStringSubmatch(response, -1)

	if len(matches) == 0 {
		return fmt.Errorf("no <OUT> sections found in the LLM response")
	}

	for _, match := range matches {
		filename := match[1]
		content := match[2]

		// Check if the filename is in the OutFiles list
		found := false
		for _, outFile := range outFiles {
			if outFile == filename {
				found = true
				break
			}
		}
		if !found {
			log.Printf("Filename %s not specified in Out: section, skipping.", filename)
			continue
		}

		// Write the content to the corresponding file
		absPath := filepath.Join(parentDir, filename)
		err := ioutil.WriteFile(absPath, []byte(content), 0644)
		if err != nil {
			log.Printf("Error writing to file %s: %v", absPath, err)
			continue
		}
		log.Printf("Updated file written to: %s", absPath)
	}
	return nil
}

// buildContextMessages builds a list of chat messages from the root to the current directory
// to provide context to the language model
func buildContextMessages(path string, watchPath string) []llm.Message {
	var messages []llm.Message
	var paths []string

	// Collect paths from root to current directory
	currentPath := path
	for {
		paths = append([]string{currentPath}, paths...)
		if currentPath == watchPath {
			// stop at the watch path
			break
		}
		parentPath := filepath.Dir(currentPath)
		if parentPath == currentPath {
			// stop at the filesystem root
			break
		}
		currentPath = parentPath
	}

	// Build messages from collected paths
	for _, p := range paths {
		if content, err := ioutil.ReadFile(filepath.Join(p, promptFn)); err == nil {
			messages = append(messages, llm.Message{
				Role:    llm.ChatMessageRoleUser,
				Content: string(content),
			})
		}
		if content, err := ioutil.ReadFile(filepath.Join(p, responseFn)); err == nil {
			messages = append(messages, llm.Message{
				Role:    llm.ChatMessageRoleAssistant,
				Content: string(content),
			})
		}
	}

	return messages
}

func getAttachmentsContent(path string) (string, error) {
	var contentBuilder strings.Builder

	files, err := ioutil.ReadDir(path)
	if err != nil {
		return "", err
	}

	for _, file := range files {
		if strings.HasSuffix(file.Name(), ".pdf.txt") {
			attachmentContent, err := ioutil.ReadFile(filepath.Join(path, file.Name()))
			if err != nil {
				return "", err
			}
			// Delimit attachments with unique XML tags
			contentBuilder.WriteString(fmt.Sprintf("<IN filename=\"%s\">\n", file.Name()))
			contentBuilder.WriteString(string(attachmentContent))
			contentBuilder.WriteString("\n</IN>\n")
		}
	}

	return contentBuilder.String(), nil
}

func getLLMResponse(messages []llm.Message, client llm.Client) (string, error) {
	ctx := context.Background()
	response, err := client.GenerateResponse(ctx, messages)
	if err != nil {
		return "", err
	}
	return response, nil
}

func handlePDFAttachment(pdfPath string, extractTextFunc func(string) (string, error)) {
	mutex.Lock()
	defer mutex.Unlock()

	text, err := extractTextFunc(pdfPath)
	if err != nil {
		log.Println("Error extracting text from PDF:", err)
		return
	}

	// Save extracted text alongside the PDF
	txtPath := pdfPath + ".txt"
	err = ioutil.WriteFile(txtPath, []byte(text), 0644)
	if err != nil {
		log.Println("Error writing extracted text:", err)
		return
	}

	log.Println("Extracted text from PDF saved to:", txtPath)
}

func extractTextFromPDF(pdfPath string) (string, error) {
	r, err := pdf.Open(pdfPath)
	if err != nil {
		return "", err
	}
	var text strings.Builder
	numPages := r.NumPage()
	for i := 1; i <= numPages; i++ {
		p := r.Page(i)
		if p.V.IsNull() {
			continue
		}
		content := p.Content()
		for _, txt := range content.Text {
			text.WriteString(txt.S + " ")
		}
	}
	return text.String(), nil
}

func createNewDecisionNode(parentPath, descriptor string) (string, error) {
	// Sanitize the descriptor to remove invalid characters
	sanitizedDescriptor := sanitizeDescriptor(descriptor)

	// Generate a unique identifier
	uuidStr := generateUUID()

	// Combine to create the directory name
	dirName := fmt.Sprintf("%s_%s", sanitizedDescriptor, uuidStr)
	newPath := filepath.Join(parentPath, dirName)

	err := os.Mkdir(newPath, 0755)
	if err != nil {
		return "", err
	}
	return newPath, nil
}

func sanitizeDescriptor(descriptor string) string {
	// Replace spaces with underscores, remove special characters
	descriptor = strings.ReplaceAll(descriptor, " ", "_")
	descriptor = strings.ReplaceAll(descriptor, "/", "_")
	descriptor = strings.ReplaceAll(descriptor, "\\", "_")
	// Add more replacements as needed
	return descriptor
}

func generateUUID() string {
	// Generate a UUID
	id := uuid.New()
	return id.String()
}

func summarizePath(path string, client llm.Client, watchPath string) {
	mutex.Lock()
	defer mutex.Unlock()

	messages := buildContextMessages(path, watchPath)
	var textBuilder strings.Builder
	for _, msg := range messages {
		textBuilder.WriteString(msg.Role + ": " + msg.Content + "\n")
	}
	text := textBuilder.String()

	summary, err := getSummary(text, client)
	if err != nil {
		log.Println("Error summarizing path:", err)
		return
	}

	summaryPath := filepath.Join(path, "summary.txt")
	err = ioutil.WriteFile(summaryPath, []byte(summary), 0644)
	if err != nil {
		log.Println("Error writing summary:", err)
	} else {
		log.Println("Summary written to:", summaryPath)
	}
}

func getSummary(text string, client llm.Client) (string, error) {
	summaryPrompt := fmt.Sprintf("Please provide a concise summary of the following conversation:\n\n%s", text)
	messages := []llm.Message{
		{
			Role:    llm.ChatMessageRoleUser,
			Content: summaryPrompt,
		},
	}
	return getLLMResponse(messages, client)
}

func updateMetrics(path string, metrics map[string]interface{}) {
	metricsPath := filepath.Join(path, "metrics.json")
	data, err := json.MarshalIndent(metrics, "", "  ")
	if err != nil {
		log.Println("Error marshalling metrics:", err)
		return
	}
	err = ioutil.WriteFile(metricsPath, data, 0644)
	if err != nil {
		log.Println("Error writing metrics:", err)
	} else {
		log.Println("Metrics updated at:", metricsPath)
	}
}
