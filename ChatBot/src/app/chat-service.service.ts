





import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { catchError, Observable, throwError } from 'rxjs';
import { AuthService } from './auth.service';

@Injectable({
  providedIn: 'root'
})
export class ChatServiceService {
  private apiUrl = 'http://localhost:8000/api';

  constructor(private http: HttpClient, private authService: AuthService) {}

  askLlama(question: string, format: string = 'text'): Observable<any> {
    const currentUser = this.authService.getCurrentUser();
    
    console.log('Current user data:', currentUser);
    
    if (!currentUser) {
      console.error('No current user found when making request');
      // You might want to handle this case better than just continuing
    }
    
    const payload = {
      question: question,
      userEmail: currentUser?.email || '',
      userRoles: currentUser?.roles || [],
      accessibleTables: currentUser?.accessibleTables || [],
      format: format
    };
    
    console.log('Sending payload to API:', payload);
    
    // Add a timeout to the request
    return this.http.post<any>(`${this.apiUrl}/ask-llama`, payload)
      .pipe(
        catchError(error => {
          console.error('Error in askLlama service:', error);
          if (error.status === 0) {
            return throwError(() => new Error('Network error. Please check if the API server is running.'));
          }
          return throwError(() => error);
        })
      );
  }

  getConversationalResponse(prompt: string): Observable<any> {
    const currentUser = this.authService.getCurrentUser();
    const payload = {
      prompt: prompt,
      userEmail: currentUser?.email || '',
      userRoles: currentUser?.roles || []
    };
    return this.http.post<any>(`${this.apiUrl}/conversational-response`, payload);
  }
  
  classifyMessage(message: string, accessibleTables: string[]): Observable<any> {
    const currentUser = this.authService.getCurrentUser();
    const payload = {
      message: message,
      accessibleTables: accessibleTables,
      userEmail: currentUser?.email || '',
      userRoles: currentUser?.roles || []
    };
    return this.http.post<any>(`${this.apiUrl}/classify-message`, payload);
  }

  getAllTables(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/tables`);
  }

  getTableSchema(tableName: string): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/table-schema/${tableName}`);
  }
}