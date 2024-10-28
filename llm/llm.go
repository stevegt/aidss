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

// Client is the interface that all LLM clients must implement.
type Client interface {
	GenerateResponse(ctx context.Context, messages []Message) (string, error)
	// Models() []string // Added Models method
}

// Provider represents an LLM provider.
type Provider interface {
	// NewClient returns a new Client instance for the given model name and apiKey.
	NewClient(modelName string, apiKey string) (Client, error)
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
func NewClient(modelName string, apiKey string) (Client, error) {
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
	return provider.NewClient(modelName, apiKey)
}

func init() {
	// Initialize and register providers
	openAIProvider := NewOpenAIProvider()
	RegisterProvider("openai", openAIProvider)
	mockProvider := NewMockProvider()
	RegisterProvider("mock", mockProvider)
}
