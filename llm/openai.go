package llm

import (
	"context"
	"errors"
	"os"

	openai "github.com/sashabaranov/go-openai"
)

type OpenAI struct {
	client *openai.Client
	model  Model
}

type OpenAIProvider struct {
	apiKey string
}

// Model struct represents a language model with its attributes
type Model struct {
	Name        string
	MaxTokens   int
	Temperature float32
}

// Map of model names to Model structs
var openAIModels = map[string]Model{
	openai.GPT4o: {
		Name:        openai.GPT4o,
		MaxTokens:   128000,
		Temperature: 0.7,
	},
	openai.O1Mini: {
		Name:        openai.O1Mini,
		MaxTokens:   128000,
		Temperature: 0.7,
	},
	openai.O1Preview: {
		Name:        openai.O1Preview,
		MaxTokens:   128000,
		Temperature: 0.7,
	},
}

// NewOpenAIProvider creates a new instance of OpenAIProvider
func NewOpenAIProvider() *OpenAIProvider {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		// Return nil if API key is not set
		return nil
	}

	return &OpenAIProvider{
		apiKey: apiKey,
	}
}

// NewClient returns a new OpenAI client for the given model
func (p *OpenAIProvider) NewClient(modelName string) (Client, error) {
	model, ok := openAIModels[modelName]
	if !ok {
		return nil, errors.New("unsupported model: " + modelName)
	}

	// Create OpenAI client
	client := openai.NewClient(p.apiKey)

	return &OpenAI{
		client: client,
		model:  model,
	}, nil
}

// Models returns the models available in OpenAI
func (p *OpenAIProvider) Models() []string {
	models := make([]string, 0, len(openAIModels))
	for modelName := range openAIModels {
		models = append(models, modelName)
	}
	return models
}

// GenerateResponse implements the Client interface
func (o *OpenAI) GenerateResponse(ctx context.Context, messages []Message) (string, error) {
	// Convert Messages to openai.ChatCompletionMessage
	var chatMessages []openai.ChatCompletionMessage
	for _, msg := range messages {
		chatMessages = append(chatMessages, openai.ChatCompletionMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	// Build the request
	req := openai.ChatCompletionRequest{
		Model:       o.model.Name,
		Messages:    chatMessages,
		MaxTokens:   o.model.MaxTokens,
		Temperature: o.model.Temperature,
	}

	// Call the OpenAI API
	resp, err := o.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", err
	}

	return resp.Choices[0].Message.Content, nil
}
