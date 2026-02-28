import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterLink } from '@angular/router';
import { ApiService, JobStatus } from '../../services/api.service';

@Component({
  selector: 'app-upload',
  imports: [CommonModule, RouterLink],
  templateUrl: './upload.component.html',
  styleUrl: './upload.component.scss',
})
export class UploadComponent implements OnInit {
  selectedFile: File | null = null;
  uploading = false;
  dragOver = false;
  jobs: JobStatus[] = [];

  constructor(private api: ApiService, private router: Router) {}

  ngOnInit() {
    console.log('[UploadPage] INIT — loading jobs');
    this.loadJobs();
  }

  loadJobs() {
    this.api.getJobs().subscribe({
      next: (jobs) => {
        console.log('[UploadPage] loaded', jobs.length, 'jobs');
        this.jobs = jobs;
      },
      error: (err) => console.warn('[UploadPage] failed to load jobs:', err),
    });
  }

  onDragOver(e: DragEvent) {
    e.preventDefault();
    this.dragOver = true;
  }

  onDragLeave() {
    this.dragOver = false;
  }

  onDrop(e: DragEvent) {
    e.preventDefault();
    this.dragOver = false;
    const file = e.dataTransfer?.files[0];
    if (file && file.type.startsWith('video/')) {
      this.selectedFile = file;
    }
  }

  onFileSelect(e: Event) {
    const input = e.target as HTMLInputElement;
    if (input.files?.length) {
      this.selectedFile = input.files[0];
    }
  }

  upload() {
    if (!this.selectedFile || this.uploading) return;
    this.uploading = true;
    this.api.uploadVideo(this.selectedFile).subscribe({
      next: (res) => {
        this.uploading = false;
        this.router.navigate(['/jobs', res.job_id, 'status']);
      },
      error: (err) => {
        this.uploading = false;
        alert('Upload failed: ' + (err.error?.detail || err.message));
      },
    });
  }

  stageLabel(stage: string): string {
    const map: Record<string, string> = {
      queued: 'Queued',
      probing: 'Probing',
      transcribing: 'Transcribing',
      generating_candidates: 'Generating Candidates',
      extracting_features: 'Extracting Features',
      scoring: 'Scoring',
      building_timeline: 'Building Timeline',
      rendering_clips: 'Rendering Clips',
      complete: 'Complete',
      error: 'Error',
    };
    return map[stage] || stage;
  }
}
