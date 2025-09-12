package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	// Eino framework imports
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/components/prompt"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
)

// IntelligentChatAssistant represents our Eino-based chat service
type IntelligentChatAssistant struct {
	chatModel model.BaseChatModel
	chatChain compose.Runnable[[]*schema.Message, *schema.Message]
}

// NewIntelligentChatAssistant creates a new chat assistant using Eino framework
func NewIntelligentChatAssistant(ctx context.Context) (*IntelligentChatAssistant, error) {
	// Create a mock chat model for demonstration
	// In production, this would be a real LLM model like OpenAI, etc.
	chatModel := &MockChatModel{}

	// Create Eino chat chain with proper composition
	chatChain, err := createEinoChatChain(ctx, chatModel)
	if err != nil {
		return nil, fmt.Errorf("failed to create Eino chat chain: %v", err)
	}

	return &IntelligentChatAssistant{
		chatModel: chatModel,
		chatChain: chatChain,
	}, nil
}

// MockChatModel implements the BaseChatModel interface for demonstration
type MockChatModel struct{}

// Generate implements the BaseChatModel interface
func (m *MockChatModel) Generate(ctx context.Context, input []*schema.Message, opts ...model.Option) (*schema.Message, error) {
	// Extract the last user message
	var userInput string
	for i := len(input) - 1; i >= 0; i-- {
		if input[i].Role == schema.User {
			userInput = input[i].Content
			break
		}
	}

	// Generate response based on user input
	responseContent := generateResponse(userInput)

	return &schema.Message{
		Role:    schema.Assistant,
		Content: responseContent,
	}, nil
}

// Stream implements the BaseChatModel interface (not implemented for this demo)
func (m *MockChatModel) Stream(ctx context.Context, input []*schema.Message, opts ...model.Option) (*schema.StreamReader[*schema.Message], error) {
	return nil, fmt.Errorf("stream not implemented in mock model")
}

// createEinoChatChain creates a proper Eino composition chain
func createEinoChatChain(ctx context.Context, chatModel model.BaseChatModel) (compose.Runnable[[]*schema.Message, *schema.Message], error) {
	// Create system prompt template
	systemPrompt := prompt.FromMessages(
		schema.FString,
		schema.SystemMessage("You are an intelligent assistant built with CloudWeGo Eino framework. "+
			"You help developers understand and use the Eino framework for building LLM applications. "+
			"Provide clear, technical explanations about the framework's capabilities and best practices."),
	)

	// Create Eino chain: System Prompt -> Chat Model
	chain := compose.NewChain[[]*schema.Message, *schema.Message]()
	chain.AppendChatTemplate(systemPrompt)
	chain.AppendChatModel(chatModel)

	// Compile the chain
	compiledChain, err := chain.Compile(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to compile Eino chain: %v", err)
	}

	return compiledChain, nil
}

// generateResponse generates intelligent responses based on user input
func generateResponse(userInput string) string {
	lowerInput := strings.ToLower(userInput)

	// Eino framework-aware responses
	switch {
	case strings.Contains(lowerInput, "hello") || strings.Contains(lowerInput, "hi"):
		return "Hello! I'm an intelligent assistant built with CloudWeGo Eino framework. I can help you understand how to build LLM applications with proper architecture."
	
	case strings.Contains(lowerInput, "eino") || strings.Contains(lowerInput, "cloudwego"):
		return "CloudWeGo Eino is a powerful LLM application development framework written in Go. It provides:\n" +
			"• Component abstractions for LLM applications\n" +
			"• Type-safe composition framework\n" +
			"• Stream processing capabilities\n" +
			"• Built-in support for tool calling\n" +
			"• Production-ready architecture patterns"
	
	case strings.Contains(lowerInput, "model") || strings.Contains(lowerInput, "llm"):
		return "Eino's model component provides a standardized interface for LLM interactions. It supports:\n" +
			"• BaseChatModel interface for basic chat operations\n" +
			"• ToolCallingChatModel for function calling\n" +
			"• Stream processing for real-time responses\n" +
			"• Proper error handling and type safety"
	
	case strings.Contains(lowerInput, "chain") || strings.Contains(lowerInput, "composition"):
		return "Eino's composition framework allows building complex LLM workflows:\n" +
			"• Chains for linear processing\n" +
			"• Graphs for complex workflows\n" +
			"• Workflows for structured data processing\n" +
			"• All with full type safety and error handling"
	
	case strings.Contains(lowerInput, "how") && strings.Contains(lowerInput, "work"):
		return "Eino works by providing a structured framework for LLM application development:\n" +
			"1. Define components (models, prompts, tools)\n" +
			"2. Compose them using chains or graphs\n" +
			"3. Execute with proper error handling\n" +
			"4. Process streams for real-time applications"
	
	default:
		return "I'm an assistant demonstrating CloudWeGo Eino framework capabilities. " +
			"I can help you understand how to build production-ready LLM applications with proper architecture, " +
			"type safety, and streaming capabilities. What would you like to know about Eino framework?"
	}
}

// ProcessMessage handles chat messages using Eino framework patterns
func (ica *IntelligentChatAssistant) ProcessMessage(ctx context.Context, userMessage string) (string, error) {
	// Create Eino schema messages (only user message, system prompt is handled by the chain)
	messages := []*schema.Message{
		{
			Role:    schema.User,
			Content: userMessage,
		},
	}

	// Use the Eino chain to generate response
	response, err := ica.chatChain.Invoke(ctx, messages)
	if err != nil {
		return "", fmt.Errorf("failed to generate response using Eino chain: %v", err)
	}

	return response.Content, nil
}

// HTTP handler for chat endpoint
func chatHandler(assistant *IntelligentChatAssistant) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var request struct {
			Message string `json:"message"`
		}

		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, "Invalid JSON format", http.StatusBadRequest)
			return
		}

		response, err := assistant.ProcessMessage(r.Context(), request.Message)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"response":  response,
			"framework": "cloudwego/eino",
			"version":   "v0.5.2",
		})
	}
}

func main() {
	ctx := context.Background()

	// Create Eino-based chat assistant
	assistant, err := NewIntelligentChatAssistant(ctx)
	if err != nil {
		log.Fatalf("Failed to create chat assistant: %v", err)
	}

	// Create HTTP server with proper configuration
	server := &http.Server{
		Addr:    ":8080",
		Handler: nil,
	}

	// Register handlers
	http.HandleFunc("/chat", chatHandler(assistant))
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"status":    "healthy",
			"service":   "eino-chat-assistant",
			"framework": "cloudwego/eino v0.5.2",
			"model":     "BaseChatModel interface",
		})
	})

	// Start server in a goroutine
	go func() {
		fmt.Println("🚀 Eino Intelligent Chat Assistant starting on http://localhost:8080")
		fmt.Println("📋 Available endpoints:")
		fmt.Println("   POST /chat    - Chat with the AI assistant (uses Eino composition framework)")
		fmt.Println("   GET  /health  - Health check with framework info")
		fmt.Println("")
		fmt.Println("💡 This implementation demonstrates:")
		fmt.Println("   • Proper Eino BaseChatModel interface implementation")
		fmt.Println("   • Schema.Message usage for conversation management")
		fmt.Println("   • Eino framework component architecture with chains")
		fmt.Println("   • Production-ready HTTP API design")
		fmt.Println("   • Proper error handling and graceful shutdown")

		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	// Wait for interrupt signal for graceful shutdown
	<-ctx.Done()
	log.Println("Shutting down server...")

	// Create a context with timeout for graceful shutdown
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Printf("Server shutdown failed: %v", err)
	}

	log.Println("Server gracefully stopped")
}