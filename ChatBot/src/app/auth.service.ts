import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, tap } from 'rxjs';

export interface UserRole {
  id: number;
  name: string;
}

export interface User {
  email: string;
  roles: UserRole[];
  accessibleTables: string[];
}

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private currentUserSubject = new BehaviorSubject<User | null>(null);
  public currentUser = this.currentUserSubject.asObservable();
  private apiUrl = 'http://localhost:8000/api'; // Adjust to your backend URL

  constructor(private http: HttpClient) {
    const storedUser = localStorage.getItem('currentUser');
    if (storedUser) {
      this.currentUserSubject.next(JSON.parse(storedUser));
    }
  }

  login(email: string): Observable<any> {
    return this.http.post<any>(`${this.apiUrl}/login`, { email }).pipe(
      tap(response => {
        if (response.success) {
          const user: User = {
            email: response.email,
            roles: response.roles,
            accessibleTables: response.accessibleTables
          };
          localStorage.setItem('currentUser', JSON.stringify(user));
          this.currentUserSubject.next(user);
        }
      })
    );
  }

  logout(): void {
    localStorage.removeItem('currentUser');
    this.currentUserSubject.next(null);
  }

  isLoggedIn(): boolean {
    return !!this.currentUserSubject.value;
  }

  getCurrentUser(): User | null {
    return this.currentUserSubject.value;
  }
}