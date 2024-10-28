package main

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/davecgh/go-spew/spew"
	"github.com/fsnotify/fsnotify"
	"github.com/stevegt/aidss/llm"

	. "github.com/stevegt/goadapt"
)

func init() {
	// Initialize and register providers for testing
	llm.RegisterProvider("mock", llm.NewMockProvider())
}

func TestCreateNewDecisionNode(t *testing.T) {
	parentDir, err := ioutil.TempDir("", "test_decision_node")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(parentDir)

	descriptor := "Test Node"
	newPath, err := createNewDecisionNode(parentDir, descriptor)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if _, err := os.Stat(newPath); os.IsNotExist(err) {
		t.Fatalf("Expected directory %s to be created", newPath)
	}

	expectedPrefix := filepath.Join(parentDir, "Test_Node_")
	if !strings.HasPrefix(newPath, expectedPrefix) {
		t.Errorf("Expected directory name to start with %s, got %s", expectedPrefix, newPath)
	}
}

func TestHandleUserMessage(t *testing.T) {
	// Set up temporary directory
	tempDir, err := ioutil.TempDir("", "test_user_message")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	// Create prompt.txt
	messageContent := "Test message"
	err = ioutil.WriteFile(filepath.Join(tempDir, "prompt.txt"), []byte(messageContent), 0644)
	if err != nil {
		t.Fatal(err)
	}

	// Create the mock client
	var errClient error
	client, errClient := llm.NewClient("mock-model")
	if errClient != nil {
		t.Fatalf("Error creating mock client: %v", errClient)
	}

	// Mock mutex
	mutex = sync.Mutex{}

	// Call handleUserMessage
	handleUserMessage(tempDir, client, tempDir)

	// Check response.txt
	responsePath := filepath.Join(tempDir, "response.txt")
	data, err := ioutil.ReadFile(responsePath)
	if err != nil {
		t.Fatalf("Expected response.txt to be created, got error: %v", err)
	}

	if string(data) != "This is a mock response." {
		t.Errorf("Expected 'This is a mock response.', got '%s'", string(data))
	}
}

func TestBuildContextMessages(t *testing.T) {
	// Set up nested directories
	rootDir, err := ioutil.TempDir("", "test_context_messages")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(rootDir)

	subDir := filepath.Join(rootDir, "subdir")
	err = os.Mkdir(subDir, 0755)
	if err != nil {
		t.Fatal(err)
	}

	// Create prompt.txt and response.txt in root
	err = ioutil.WriteFile(filepath.Join(rootDir, "prompt.txt"), []byte("Root message"), 0644)
	if err != nil {
		t.Fatal(err)
	}
	err = ioutil.WriteFile(filepath.Join(rootDir, "response.txt"), []byte("Root response"), 0644)
	if err != nil {
		t.Fatal(err)
	}

	// Create prompt.txt and response.txt in subdir
	err = ioutil.WriteFile(filepath.Join(subDir, "prompt.txt"), []byte("Subdir message"), 0644)
	if err != nil {
		t.Fatal(err)
	}
	err = ioutil.WriteFile(filepath.Join(subDir, "response.txt"), []byte("Subdir response"), 0644)
	if err != nil {
		t.Fatal(err)
	}

	// Call buildContextMessages
	messages := buildContextMessages(subDir, rootDir)

	// Expected messages
	expectedMessages := []llm.Message{
		{Role: llm.ChatMessageRoleUser, Content: "Root message"},
		{Role: llm.ChatMessageRoleAssistant, Content: "Root response"},
		{Role: llm.ChatMessageRoleUser, Content: "Subdir message"},
		{Role: llm.ChatMessageRoleAssistant, Content: "Subdir response"},
	}

	if len(messages) != len(expectedMessages) {
		spew.Dump(messages)
		t.Fatalf("Expected %d messages, got %d", len(expectedMessages), len(messages))
	}

	for i, msg := range messages {
		if msg.Role != expectedMessages[i].Role || msg.Content != expectedMessages[i].Content {
			t.Errorf("Message %d expected %+v, got %+v", i, expectedMessages[i], msg)
		}
	}
}

func TestGetAttachmentsContent(t *testing.T) {
	// Set up temporary directory
	tempDir, err := ioutil.TempDir("", "test_attachments")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	// Create attachment.pdf.txt
	attachmentContent := "Extracted text from PDF"
	err = ioutil.WriteFile(filepath.Join(tempDir, "attachment.pdf.txt"), []byte(attachmentContent), 0644)
	if err != nil {
		t.Fatal(err)
	}

	// Call getAttachmentsContent
	content, err := getAttachmentsContent(tempDir)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	expectedContent := "<Attachment filename=\"attachment.pdf.txt\">\n" + attachmentContent + "\n</Attachment>\n"
	if content != expectedContent {
		t.Errorf("Expected content:\n%s\nGot:\n%s", expectedContent, content)
	}
}

func TestGetLLMResponse(t *testing.T) {
	// Create the mock client
	var errClient error
	client, errClient := llm.NewClient("mock-model")
	if errClient != nil {
		t.Fatalf("Error creating mock client: %v", errClient)
	}

	// Mock messages
	messages := []llm.Message{
		{Role: llm.ChatMessageRoleUser, Content: "Hello"},
	}

	// Call getLLMResponse
	response, err := getLLMResponse(messages, client)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if response != "This is a mock response." {
		t.Errorf("Expected 'This is a mock response.', got '%s'", response)
	}
}

func TestHandlePDFAttachment(t *testing.T) {
	// Set up temporary directory
	tempDir, err := ioutil.TempDir("", "test_pdf_attachment")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	// Create a fake PDF file (since we're not testing PDF parsing here)
	pdfPath := filepath.Join(tempDir, "attachment.pdf")
	err = ioutil.WriteFile(pdfPath, []byte("%PDF-1.4 Fake PDF content"), 0644)
	if err != nil {
		t.Fatal(err)
	}

	// Mock extractTextFromPDF function
	mockExtractText := func(pdfPath string) (string, error) {
		return "Extracted text", nil
	}

	// Call handlePDFAttachment
	handlePDFAttachment(pdfPath, mockExtractText)

	// Check if attachment.pdf.txt is created
	txtPath := pdfPath + ".txt"
	data, err := ioutil.ReadFile(txtPath)
	if err != nil {
		t.Fatalf("Expected %s to be created, got error: %v", txtPath, err)
	}

	if string(data) != "Extracted text" {
		t.Errorf("Expected 'Extracted text', got '%s'", string(data))
	}
}

func TestExtractTextFromPDF(t *testing.T) {
	// Since testing actual PDF extraction is complex, we'll test error handling
	// Attempt to extract text from a non-existent PDF
	_, err := extractTextFromPDF("non_existent.pdf")
	if err == nil {
		t.Errorf("Expected error when extracting from non-existent PDF")
	}
}

func TestSummarizePath(t *testing.T) {
	// Set up temporary directory
	tempDir, err := ioutil.TempDir("", "test_summarize")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	// Create conversation files
	err = ioutil.WriteFile(filepath.Join(tempDir, "prompt.txt"), []byte("User message"), 0644)
	if err != nil {
		t.Fatal(err)
	}
	err = ioutil.WriteFile(filepath.Join(tempDir, "response.txt"), []byte("LLM response"), 0644)
	if err != nil {
		t.Fatal(err)
	}

	// Create the mock client
	var errClient error
	client, errClient := llm.NewClient("mock-model")
	if errClient != nil {
		t.Fatalf("Error creating mock client: %v", errClient)
	}

	// Call summarizePath
	summarizePath(tempDir, client, tempDir)

	// Check if summary.txt is created
	summaryPath := filepath.Join(tempDir, "summary.txt")
	data, err := ioutil.ReadFile(summaryPath)
	if err != nil {
		t.Fatalf("Expected summary.txt to be created, got error: %v", err)
	}

	if string(data) != "This is a mock response." {
		t.Errorf("Expected 'This is a mock response.', got '%s'", string(data))
	}
}

func TestUpdateMetrics(t *testing.T) {
	// Set up temporary directory
	tempDir, err := ioutil.TempDir("", "test_metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	// Define metrics
	metrics := map[string]interface{}{
		"completeness":        0.9,
		"logical_consistency": 0.95,
	}

	// Call updateMetrics
	updateMetrics(tempDir, metrics)

	// Check if metrics.json is created
	metricsPath := filepath.Join(tempDir, "metrics.json")
	data, err := ioutil.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("Expected metrics.json to be created, got error: %v", err)
	}

	var readMetrics map[string]interface{}
	err = json.Unmarshal(data, &readMetrics)
	if err != nil {
		t.Fatalf("Error unmarshalling metrics.json: %v", err)
	}

	if readMetrics["completeness"] != metrics["completeness"] {
		t.Errorf("Expected completeness %v, got %v", metrics["completeness"], readMetrics["completeness"])
	}

	if readMetrics["logical_consistency"] != metrics["logical_consistency"] {
		t.Errorf("Expected logical_consistency %v, got %v", metrics["logical_consistency"], readMetrics["logical_consistency"])
	}
}

func TestAddWatcherRecursive(t *testing.T) {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		t.Fatal(err)
	}
	defer watcher.Close()

	// Set up nested directories
	rootDir, err := ioutil.TempDir("", "test_watcher")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(rootDir)

	subDir := filepath.Join(rootDir, "subdir")
	err = os.Mkdir(subDir, 0755)
	if err != nil {
		t.Fatal(err)
	}

	// Call addWatcherRecursive
	err = addWatcherRecursive(watcher, rootDir)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	// Check if both directories are being watched
	list := watcher.WatchList()
	for _, want := range []string{rootDir, subDir} {
		found := false
		for _, got := range list {
			if got == want {
				found = true
				break
			}
		}
		Tassert(t, found, "Expected %s to be in watch list", want)
	}
}

func TestGetSummary(t *testing.T) {
	// Create the mock client
	var errClient error
	client, errClient := llm.NewClient("mock-model")
	if errClient != nil {
		t.Fatalf("Error creating mock client: %v", errClient)
	}

	// Call getSummary
	text := "Conversation text"
	summary, err := getSummary(text, client)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if summary != "This is a mock response." {
		t.Errorf("Expected 'This is a mock response.', got '%s'", summary)
	}
}

func TestErrorHandlingInHandleUserMessage(t *testing.T) {
	// Use a non-existent directory
	nonExistentPath := "non_existent_dir"

	// Create the mock client
	var errClient error
	client, errClient := llm.NewClient("mock-model")
	if errClient != nil {
		t.Fatalf("Error creating mock client: %v", errClient)
	}

	// Call handleUserMessage
	handleUserMessage(nonExistentPath, client, nonExistentPath)

	// Expect no panic and error to be logged
}

func TestErrorHandlingInGetAttachmentsContent(t *testing.T) {
	// Use a non-existent directory
	nonExistentPath := "non_existent_dir"

	// Call getAttachmentsContent
	_, err := getAttachmentsContent(nonExistentPath)
	if err == nil {
		t.Errorf("Expected error when reading attachments from non-existent directory")
	}
}

func TestErrorHandlingInUpdateMetrics(t *testing.T) {
	// Use a read-only directory
	tempDir, err := ioutil.TempDir("", "test_metrics_readonly")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	// Make directory read-only
	err = os.Chmod(tempDir, 0444)
	if err != nil {
		t.Fatal(err)
	}

	// Define metrics
	metrics := map[string]interface{}{
		"test_metric": 1.0,
	}

	// Call updateMetrics
	updateMetrics(tempDir, metrics)

	// Expect error to be logged
}
