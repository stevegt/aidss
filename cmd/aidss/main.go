package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
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
	watchPath string
	modelName string
	client    llm.Client
	mutex     sync.Mutex // To handle concurrent access

	promptFn   = "prompt.txt"
	responseFn = "response.txt"
)

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
			startDaemon()
		},
	}

	// Make sure usage includes model names
	modelUsage := fmt.Sprintf("Model to use (%s)", strings.Join(models, ", "))

	// Define flags
	rootCmd.Flags().StringVarP(&watchPath, "path", "p", ".", "Path to watch")
	rootCmd.Flags().StringVarP(&modelName, "model", "m", models[0], modelUsage)

	// Execute the root command
	if err := rootCmd.Execute(); err != nil {
		log.Fatal(err)
	}
}

// startDaemon starts the decision tool daemon. The daemon watches the file system for changes
// and responds to user messages and attachments.
func startDaemon() {
	var err error

	// Set up the LLM client based on the model name
	client, err = llm.NewClient(modelName)
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
						handleUserMessage(filepath.Dir(event.Name))
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

// handleUserMessage handles a user message by generating a response from the language model
func handleUserMessage(path string) {
	mutex.Lock()
	defer mutex.Unlock()

	messagePath := filepath.Join(path, promptFn)
	message, err := ioutil.ReadFile(messagePath)
	if err != nil {
		log.Println("Error reading user message:", err)
		return
	}

	// Build context messages
	contextMessages := buildContextMessages(path)

	// Append the new message
	contextMessages = append(contextMessages, llm.Message{
		Role:    llm.ChatMessageRoleUser,
		Content: string(message),
	})

	// Check if any attachments need to be included
	attachmentsContent, err := getAttachmentsContent(path)
	if err != nil {
		log.Println("Error reading attachments:", err)
		return
	}

	if attachmentsContent != "" {
		// Add the attachments content to the system prompt
		contextMessages = append([]llm.Message{
			{
				Role:    llm.ChatMessageRoleSystem,
				Content: "The following attachments are included:\n" + attachmentsContent,
			},
		}, contextMessages...)
	}

	response, err := getLLMResponse(contextMessages)
	if err != nil {
		log.Println("Error getting LLM response:", err)
		return
	}

	responsePath := filepath.Join(path, responseFn)
	err = ioutil.WriteFile(responsePath, []byte(response), 0644)
	if err != nil {
		log.Println("Error writing LLM response:", err)
	}

	log.Println("LLM response written to:", responsePath)
}

// buildContextMessages builds a list of chat messages from the root to the current directory
// to provide context to the language model
func buildContextMessages(path string) []llm.Message {
	var messages []llm.Message
	var paths []string

	// Collect paths from root to current directory
	currentPath := path
	for {
		paths = append([]string{currentPath}, paths...)
		parentPath := filepath.Dir(currentPath)
		if parentPath == currentPath || parentPath == "." {
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
			contentBuilder.WriteString("Attachment: " + file.Name() + "\n")
			contentBuilder.WriteString(string(attachmentContent) + "\n")
		}
	}

	return contentBuilder.String(), nil
}

func getLLMResponse(messages []llm.Message) (string, error) {
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

func summarizePath(path string) {
	mutex.Lock()
	defer mutex.Unlock()

	messages := buildContextMessages(path)
	var textBuilder strings.Builder
	for _, msg := range messages {
		textBuilder.WriteString(msg.Role + ": " + msg.Content + "\n")
	}
	text := textBuilder.String()

	summary, err := getSummary(text)
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

func getSummary(text string) (string, error) {
	summaryPrompt := fmt.Sprintf("Please provide a concise summary of the following conversation:\n\n%s", text)
	messages := []llm.Message{
		{
			Role:    llm.ChatMessageRoleUser,
			Content: summaryPrompt,
		},
	}
	return getLLMResponse(messages)
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
