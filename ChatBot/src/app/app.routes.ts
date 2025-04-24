import { Routes } from '@angular/router';
import { ChatbotComponent } from './chatbot/chatbot.component';
import { LoginComponent } from './login/login.component';
import { AuthGuard } from './auth.guard'; // Create this guard

export const routes: Routes = [
  { path: '', redirectTo: '/login', pathMatch: 'full' }, // Default route to login
  { path: 'login', component: LoginComponent },
  { path: 'chat', component: ChatbotComponent, canActivate: [AuthGuard] }, // Protect chatbot route
  { path: '**', redirectTo: '/login' } // Wildcard route redirects to login
];