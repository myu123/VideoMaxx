import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router } from '@angular/router';
import { ApiService, JobStatus } from '../../services/api.service';

@Component({
  selector: 'app-status',
  imports: [CommonModule],
  templateUrl: './status.component.html',
  styleUrl: './status.component.scss',
})
export class StatusComponent implements OnInit, OnDestroy {
  jobId = '';
  status: JobStatus | null = null;
  elapsed = 0;
  connectionError = '';
  pollDebug = '';
  private interval: any;
  private timerInterval: any;
  private startTime = Date.now();

  stages = [
    { key: 'probing', label: 'Analyzing video file', icon: '1' },
    { key: 'transcribing', label: 'Transcribing audio (Whisper GPU)', icon: '2' },
    { key: 'generating_candidates', label: 'Generating candidate segments', icon: '3' },
    { key: 'extracting_features', label: 'Extracting features (audio + video + text)', icon: '4' },
    { key: 'scoring', label: 'ML scoring & explainability', icon: '5' },
    { key: 'building_timeline', label: 'Building engagement timeline', icon: '6' },
    { key: 'rendering_clips', label: 'Rendering clips with captions', icon: '7' },
  ];

  constructor(
    private route: ActivatedRoute,
    private api: ApiService,
    private router: Router,
  ) {}

  ngOnInit() {
    this.jobId = this.route.snapshot.paramMap.get('jobId') || '';
    console.log('[StatusPage] INIT — jobId =', this.jobId);
    this.pollDebug = `Job: ${this.jobId} | Polling...`;
    this.startTime = Date.now();

    if (!this.jobId) {
      this.connectionError = 'No job ID found in URL';
      return;
    }

    this.poll();
    this.interval = setInterval(() => this.poll(), 2000);
    this.timerInterval = setInterval(() => {
      this.elapsed = Math.floor((Date.now() - this.startTime) / 1000);
    }, 1000);
  }

  ngOnDestroy() {
    if (this.interval) clearInterval(this.interval);
    if (this.timerInterval) clearInterval(this.timerInterval);
  }

  poll() {
    const url = `/api/jobs/${this.jobId}/status`;
    console.log('[StatusPage] polling:', url);

    this.api.getJobStatus(this.jobId).subscribe({
      next: (s) => {
        console.log('[StatusPage] GOT DATA:', s.stage, s.progress, '%');
        this.pollDebug = `Stage: ${s.stage} | Progress: ${s.progress}% | Logs: ${s.logs?.length ?? 0}`;
        this.status = { ...s };
        this.connectionError = '';

        if (s.stage === 'complete') {
          clearInterval(this.interval);
          clearInterval(this.timerInterval);
          setTimeout(() => this.router.navigate(['/jobs', this.jobId, 'results']), 1500);
        }
        if (s.stage === 'error') {
          clearInterval(this.interval);
          clearInterval(this.timerInterval);
        }
      },
      error: (err) => {
        console.error('[StatusPage] POLL ERROR:', err.status, err.statusText, err);
        this.pollDebug = `ERROR: ${err.status} ${err.statusText}`;
        if (err.status === 0) {
          this.connectionError = 'Cannot reach backend — is start_backend.bat running?';
        } else if (err.status === 404) {
          this.connectionError = 'Job not found. Backend may have restarted — try uploading again.';
          clearInterval(this.interval);
        } else {
          this.connectionError = `API error ${err.status}: ${err.statusText || 'Unknown'}`;
        }
      },
    });
  }

  stageIndex(): number {
    if (!this.status) return -1;
    return this.stages.findIndex(s => s.key === this.status!.stage);
  }

  isStageComplete(i: number): boolean {
    if (this.status?.stage === 'complete') return true;
    return i < this.stageIndex();
  }

  isStageActive(i: number): boolean {
    if (this.status?.stage === 'complete') return false;
    return i === this.stageIndex();
  }

  formatElapsed(seconds: number): string {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return m > 0 ? `${m}m ${s}s` : `${s}s`;
  }

  formatDuration(seconds: number): string {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  }

  formatFileSize(mb: number): string {
    if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
    return `${mb} MB`;
  }

  stageDescription(): string {
    if (!this.status) return 'Starting pipeline...';
    const map: Record<string, string> = {
      queued: 'Waiting to start...',
      probing: 'Reading video metadata with ffprobe...',
      transcribing: 'Running Whisper speech-to-text on GPU...',
      generating_candidates: 'Merging transcript into 15-60s highlight windows...',
      extracting_features: 'Analyzing motion, faces, color, audio, and text...',
      scoring: 'Running ML model to predict engagement scores...',
      building_timeline: 'Interpolating scores into engagement curve...',
      rendering_clips: 'Cutting clips and burning subtitles via ffmpeg...',
      complete: 'All done! Redirecting to results...',
      error: 'Processing failed.',
    };
    return map[this.status.stage] || this.status.stage;
  }
}
