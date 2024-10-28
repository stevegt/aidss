package llm

import (
	"context"
	"fmt"
	"sync"
)

// Message represents a chat message.
type Message struct {
	Role    string // e.g., "user", "assistant", "system"
	Content string
}

// Define constants for message roles
const (
	ChatMessageRoleUser      = "user"
	ChatMessageRoleAssistant = "assistant"
	ChatMessageRoleSystem    = "system"
)

// Client is the interface that all LLM clients must implement.
type Client interface {
	GenerateResponse(ctx context.Context, messages []Message) (string, error)
}

// Provider represents an LLM provider.
type Provider interface {
	// NewClient returns a new Client instance for the given model name
	NewClient(modelName string) (Client, error)
	// Models returns a list of model names supported by this provider.
	Models() []string
}

var (
	// Mutex for thread-safe access to the provider registry.
	registryMutex sync.Mutex
	// Map of provider names to Provider instances.
	providers = make(map[string]Provider)
	// Map of model names to provider names.
	modelToProvider = make(map[string]string)
)

// RegisterProvider registers a provider with the llm package.
func RegisterProvider(providerName string, provider Provider) {
	registryMutex.Lock()
	defer registryMutex.Unlock()

	providers[providerName] = provider
	for _, model := range provider.Models() {
		modelToProvider[model] = providerName
	}
}

// Models returns all models from all registered providers.
func Models() []string {
	registryMutex.Lock()
	defer registryMutex.Unlock()

	models := make([]string, 0, len(modelToProvider))
	for model := range modelToProvider {
		models = append(models, model)
	}
	return models
}

// NewClient returns a Client for the given model name.
func NewClient(modelName string) (Client, error) {
	registryMutex.Lock()
	defer registryMutex.Unlock()

	providerName, ok := modelToProvider[modelName]
	if !ok {
		return nil, fmt.Errorf("model %s not supported", modelName)
	}
	provider, ok := providers[providerName]
	if !ok {
		return nil, fmt.Errorf("provider %s not found for model %s", providerName, modelName)
	}
	return provider.NewClient(modelName)
}

func RegisterProviders() {
	// Initialize and register providers
	openAIProvider := NewOpenAIProvider()
	if openAIProvider != nil {
		RegisterProvider("openai", openAIProvider)
	}
	// more providers can be added here

	// Register a mock provider for testing
	mockProvider := NewMockProvider()
	RegisterProvider("mock", mockProvider)
}
