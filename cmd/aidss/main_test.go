package main

import (
	"encoding/json"
	"fmt"
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
	promptContent := `In:
	 test_in.txt
Out:
	 test_out.txt
Sysmsg: Test Sys Message
	 Additional sys message line.

Test prompt text.`

	err = ioutil.WriteFile(filepath.Join(tempDir, "prompt.txt"), []byte(promptContent), 0644)
	if err != nil {
		t.Fatal(err)
	}

	// Create test_in.txt
	inFilePath := filepath.Join(tempDir, "test_in.txt")
	err = ioutil.WriteFile(inFilePath, []byte("Content of test_in.txt"), 0644)
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

	// Check prompt-full.txt
	promptFullPath := filepath.Join(tempDir, "prompt-full.txt")
	promptData, err := ioutil.ReadFile(promptFullPath)
	if err != nil {
		t.Fatalf("Expected prompt-full.txt to be created, got error: %v", err)
	}

	expectedPromptFullContent := `User: Test prompt text.

The following files are attached:
<IN filename="test_in.txt">
Content of test_in.txt
</IN>

`

	if !strings.Contains(string(promptData), expectedPromptFullContent) {
		t.Errorf("Expected prompt-full.txt to contain '%s', got '%s'", expectedPromptFullContent, string(promptData))
	}
}

func TestParsePromptFile(t *testing.T) {
	promptContent := `In:
	 file1.txt file2.txt
Out:
	 output1.txt
	 output2.txt
Sysmsg: This is a system message
	 Continued sys message.

This is the prompt text.`

	tempFile, err := ioutil.TempFile("", "prompt_*.txt")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tempFile.Name())

	_, err = tempFile.WriteString(promptContent)
	if err != nil {
		t.Fatal(err)
	}

	prompt, err := parsePromptFile(tempFile.Name())
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	expectedInFiles := []string{"file1.txt", "file2.txt"}
	expectedOutFiles := []string{"output1.txt", "output2.txt"}
	expectedSysMsg := "This is a system message Continued sys message."
	expectedPromptText := "This is the prompt text."

	if !equalStringSlices(prompt.InFiles, expectedInFiles) {
		t.Errorf("Expected InFiles %v, got %v", expectedInFiles, prompt.InFiles)
	}
	if !equalStringSlices(prompt.OutFiles, expectedOutFiles) {
		t.Errorf("Expected OutFiles %v, got %v", expectedOutFiles, prompt.OutFiles)
	}
	if prompt.SysMsg != expectedSysMsg {
		t.Errorf("Expected SysMsg '%s', got '%s'", expectedSysMsg, prompt.SysMsg)
	}
	if prompt.PromptText != expectedPromptText {
		t.Errorf("Expected PromptText '%s', got '%s'", expectedPromptText, prompt.PromptText)
	}
}

func equalStringSlices(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	m := make(map[string]int)
	for _, x := range a {
		m[x]++
	}
	for _, x := range b {
		if m[x] == 0 {
			return false
		}
		m[x]--
	}
	return true
}

func TestProcessLLMResponse(t *testing.T) {
	response := `<OUT filename="output1.txt">
Content for output1
</OUT>

<OUT filename="output2.txt">
Content for output2
</OUT>`

	tempDir, err := ioutil.TempDir("", "test_process_response")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	outFiles := []string{"output1.txt", "output2.txt"}

	err = processLLMResponse(response, outFiles, tempDir)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	// Check if files are written correctly
	for _, fname := range outFiles {
		absPath := filepath.Join(tempDir, fname)
		data, err := ioutil.ReadFile(absPath)
		if err != nil {
			t.Fatalf("Expected file %s to be created, got error: %v", absPath, err)
		}
		expectedContent := fmt.Sprintf("Content for %s", strings.TrimSuffix(fname, ".txt"))
		if string(data) != expectedContent {
			t.Errorf("Expected content '%s', got '%s'", expectedContent, string(data))
		}
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

	// Create prompt-full.txt and response.txt in root
	err = ioutil.WriteFile(filepath.Join(rootDir, "prompt-full.txt"), []byte("Root full prompt"), 0644)
	if err != nil {
		t.Fatal(err)
	}
	err = ioutil.WriteFile(filepath.Join(rootDir, "response.txt"), []byte("Root response"), 0644)
	if err != nil {
		t.Fatal(err)
	}

	// Create prompt-full.txt and response.txt in subdir
	err = ioutil.WriteFile(filepath.Join(subDir, "prompt-full.txt"), []byte("Subdir full prompt"), 0644)
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
		{Role: llm.ChatMessageRoleUser, Content: "Root full prompt"},
		{Role: llm.ChatMessageRoleAssistant, Content: "Root response"},
		{Role: llm.ChatMessageRoleUser, Content: "Subdir full prompt"},
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

// Remaining tests unchanged...
// ...
