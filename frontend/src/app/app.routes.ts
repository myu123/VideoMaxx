import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('./pages/upload/upload.component').then(m => m.UploadComponent),
  },
  {
    path: 'jobs/:jobId/status',
    loadComponent: () => import('./pages/status/status.component').then(m => m.StatusComponent),
  },
  {
    path: 'jobs/:jobId/label',
    loadComponent: () => import('./pages/label/label.component').then(m => m.LabelComponent),
  },
  {
    path: 'jobs/:jobId/results',
    loadComponent: () => import('./pages/results/results.component').then(m => m.ResultsComponent),
  },
  {
    path: 'train',
    loadComponent: () => import('./pages/train/train.component').then(m => m.TrainComponent),
  },
  {
    path: '**',
    redirectTo: '',
  },
];
