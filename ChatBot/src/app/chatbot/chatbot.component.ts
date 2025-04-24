import { CommonModule } from '@angular/common';
import { HttpClientModule, HttpErrorResponse } from '@angular/common/http';
import { Component, ElementRef, ViewChild, AfterViewInit, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ChatServiceService } from '../chat-service.service';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { AuthService, User, UserRole } from '../auth.service';
import { Router } from '@angular/router';
  
interface Message {
  name: string;
  message: string;
  isTable?: boolean;
}

interface TableInfo {
  name: string;
  schema?: string;
}

@Component({
  selector: 'app-chatbot',
  standalone: true,
  imports: [CommonModule, FormsModule, HttpClientModule],
  templateUrl: './chatbot.component.html',
  styleUrl: './chatbot.component.css',
  providers: []
})
export class ChatbotComponent implements AfterViewInit, OnInit {
  @ViewChild('messageInput') messageInput!: ElementRef<HTMLInputElement>;
  @ViewChild('chatMessages') chatMessages!: ElementRef<HTMLDivElement>;
  
  messages: Message[] = [];
  inputMessage: string = '';
  isOpen: boolean = false;
  isLoading: boolean = false;
  currentUser: User | null = null;
  accessibleTables: string[] = [];
  tableSchemas: Map<string, string> = new Map();
  selectedTable: string = '';
  
  constructor(
    private chatService: ChatServiceService,
    private sanitizer: DomSanitizer,
    private authService: AuthService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.authService.currentUser.subscribe(user => {
      if (user) {
        this.currentUser = user;
        this.accessibleTables = user.accessibleTables;
        const roleName = user.roles.length > 0 ? user.roles[0].name : 'User';
        
        // Add welcome message
        this.messages.push({
          name: 'ArcKa',
          message: `Welcome ${roleName}! I'm ArcKa, your assistant. How can I help you today?`
        });
        
        // Set the default table if available
        if (this.accessibleTables.length > 0) {
          this.selectedTable = this.accessibleTables[0];
          console.log(`Default table set to: ${this.selectedTable}`);
        } else {
          console.log('No accessible tables available');
        }
      } else {
        this.router.navigate(['/login']);
      }
    });
  }

  ngAfterViewInit(): void {
    // Only focus the input if the chatbox is open
    if (this.messageInput && this.isOpen) {
      this.messageInput.nativeElement.focus();
    }
  }

  toggleChatbox(): void {
    this.isOpen = !this.isOpen;
    console.log("Chatbox toggled: ", this.isOpen);
    
    if (this.isOpen && this.messageInput) {
      setTimeout(() => {
        this.messageInput.nativeElement.focus();
        this.scrollToBottom(); 
      }, 100);
    }
  }

  handleKeyUp(event: KeyboardEvent): void {
    if (event.key === 'Enter') {
      this.sendMessage();
    }
  }
  
  logout(): void {
    this.authService.logout();
    this.router.navigate(['/login']);
  }
  
  sanitizeHtml(html: string): SafeHtml {
    return this.sanitizer.bypassSecurityTrustHtml(html);
  }

  detectTableInMessage(message: string): string | null {
    const lowerMessage = message.toLowerCase();
    
    // First check exact table names (case insensitive)
    for (const table of this.accessibleTables) {
      if (lowerMessage.includes(table.toLowerCase())) {
        return table;
      }
    }
    
    return null;
  }

  sendMessage(): void {
    if (this.inputMessage.trim() === '' || this.isLoading) {
      return;
    }

    const userMessage = this.inputMessage;
    this.messages.push({ name: 'User', message: userMessage });
    this.inputMessage = '';
    
    this.scrollToBottom();
    this.isLoading = true;
    
    // First, classify the message using the LLM to determine if it's conversational or a database query
    this.chatService.classifyMessage(userMessage, this.accessibleTables).subscribe({
      next: (classification) => {
        if (classification.success && classification.type === 'conversational') {
          // Handle as conversational message
          const prompt = `The user said: "${userMessage}"
          
You are a helpful database assistant name ArcKa. Respond in a friendly, conversational way.
Keep your response brief and natural.`;
          
          this.chatService.getConversationalResponse(prompt).subscribe({
            next: (response) => {
              this.isLoading = false;
              if (response.success) {
                this.messages.push({ 
                  name: 'ArcKa', 
                  message: response.message
                });
              } else {
                this.messages.push({ 
                  name: 'ArcKa', 
                  message: "I'm sorry, I couldn't process your message. How can I help you with your data queries?"
                });
              }
              this.scrollToBottom();
            },
            error: (error) => {
              this.handleResponseError(error);
            }
          });
        } else {
          // Handle as database query
          this.handleDatabaseQuery(userMessage);
        }
      },
      error: (error) => {
        // If classification fails, default to treating it as a database query
        console.error('Classification error:', error);
        this.handleDatabaseQuery(userMessage);
      }
    });
  }
  
  handleDatabaseQuery(userMessage: string): void {
    const formatPreference = userMessage.toLowerCase().includes("tabular format") || 
                           userMessage.toLowerCase().includes("table format") ? 
                           "table" : "text";
                           
    // Detect table in the message
    let tableToUse = this.detectTableInMessage(userMessage);
    
    // Check if we have accessible tables and set default if needed
    if (!tableToUse && this.accessibleTables && this.accessibleTables.length > 0) {
      if (this.selectedTable && this.accessibleTables.includes(this.selectedTable)) {
        tableToUse = this.selectedTable;
      } else {
        tableToUse = this.accessibleTables[0];
      }
      console.log(`Using default table: ${tableToUse}`);
    }
    
    // If still no table, show an error message
    if (!tableToUse) {
      this.isLoading = false;
      this.messages.push({ 
        name: 'ArcKa', 
        message: "I couldn't determine which table to query. Please specify a table in your question or check your permissions."
      });
      this.scrollToBottom();
      return;
    }
    
    // Prepare the enhanced message with table info
    const enhancedMessage = `[Table: ${tableToUse}] ${userMessage}`;
    
    console.log('Sending request with payload:', {
      question: enhancedMessage,
      format: formatPreference,
      userEmail: this.currentUser?.email,
      userRoles: this.currentUser?.roles,
      accessibleTables: this.accessibleTables
    });
    
    // Send the query to the backend
    this.chatService.askLlama(enhancedMessage, formatPreference).subscribe({
      next: (response) => {
        this.isLoading = false;
        console.log('Response received:', response);
        
        if (response && response.success) {
          if (response.format === "table" && response.results && Array.isArray(response.results) && response.results.length > 0) {
            try {
              // If the response is in table format
              let tableHtml = this.formatResultsAsTable(response.results);
              this.messages.push({ 
                name: 'ArcKa', 
                message: tableHtml,
                isTable: true
              });
            } catch (err) {
              console.error('Error formatting table:', err);
              this.messages.push({ 
                name: 'ArcKa', 
                message: 'I had trouble formatting the results as a table.'
              });
            }
          } else if (response.message) {
            // If the response is in natural language text format
            this.messages.push({ 
              name: 'ArcKa', 
              message: response.message
            });
          } else {
            // Fallback message if no results or message
            this.messages.push({ 
              name: 'ArcKa', 
              message: 'No results found for your query.'
            });
          }
        } else {
          // Handle error cases with success: false
          let errorMessage = "I'm sorry, I wasn't able to process your query. There may be an issue with accessing Azure OpenAI or with the database connection.";
          
          // Check if there's a valid error message in the response
          if (response && response.error && typeof response.error === 'string' && response.error.trim() !== '') {
            if (response.error.includes('permission')) {
              errorMessage = `Sorry, you don't have permission to access this data. As a ${this.currentUser?.roles[0]?.name || 'User'}, you can only query certain tables.`;
            } else if (response.error.includes('specify a table')) {
              const availableTables = this.accessibleTables.slice(0, 3).join(', ');
              const moreTables = this.accessibleTables.length > 3 ? '...' : '';
              errorMessage = `I need to know which table to query. You can access: ${availableTables}${moreTables}. Please specify a table in your question.`;
            } else {
              errorMessage = `I'm sorry, I encountered an error: ${response.error}. Could you try rephrasing your question?`;
            }
          } else {
            // Add helpful suggestions for common issues
            errorMessage += " Please try:\n1. Using a simpler query\n2. Checking if Azure OpenAI service is properly configured\n3. Ensuring your database connection is active";
          }
          
          console.error('Error:', response?.error || 'Unknown error');
          this.messages.push({ name: 'ArcKa', message: errorMessage });
        }
        
        this.scrollToBottom();
      },
      error: (error: HttpErrorResponse) => {
        console.error('HTTP Error:', error);
        this.handleResponseError(error);
      }
    });
  }
  
  handleResponseError(error: any): void {
    this.isLoading = false;
    
    let errorMessage = "Sorry, I couldn't process your request. Please try again.";
    
    // Check if it's an HTTP error with details
    if (error instanceof HttpErrorResponse) {
      console.error(`HTTP Error ${error.status}: ${error.statusText}`);
      console.error('Error details:', error.error);
      
      if (error.status === 0) {
        errorMessage = "Network error. Please check your connection to the server.";
      } else if (error.status === 401 || error.status === 403) {
        errorMessage = "You don't have permission to access this information.";
      } else if (error.status === 404) {
        errorMessage = "The requested resource was not found.";
      } else if (error.status >= 500) {
        errorMessage = "The server encountered an error. Please try again later.";
      }
    } else {
      console.error('Unknown error:', error);
    }
    
    this.messages.push({ 
      name: 'ArcKa', 
      message: errorMessage
    });
    this.scrollToBottom();
  }

  formatResultsAsTable(results: any[]): string {
    if (!Array.isArray(results) || results.length === 0) {
      return 'No results found.';
    }
    
    try {
      const keys = Object.keys(results[0] || {});
      if (keys.length === 0) {
        return 'The result does not contain any data columns.';
      }
      
      let tableHtml = '<table class="sql-table">';
      tableHtml += '<tr>';
      keys.forEach(key => {
        tableHtml += `<th>${key}</th>`;
      });
      tableHtml += '</tr>';
      
      results.forEach(row => {
        tableHtml += '<tr>';
        keys.forEach(key => {
          const cellValue = row[key];
          const displayValue = cellValue !== null && cellValue !== undefined ? 
            (typeof cellValue === 'object' ? JSON.stringify(cellValue) : cellValue) : '';
          tableHtml += `<td>${displayValue}</td>`;
        });
        tableHtml += '</tr>';
      });
      
      tableHtml += '</table>';
      return tableHtml;
    } catch (err) {
      console.error('Error formatting table:', err);
      return 'Error formatting results as table. Please try a different format.';
    }
  }

  scrollToBottom(): void {
    setTimeout(() => {
      if (this.chatMessages) {
        const element = this.chatMessages.nativeElement;
        element.scrollTop = element.scrollHeight;
      }
    }, 0);
  }
}