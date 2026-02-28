import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface LogEntry {
  time: number;
  message: string;
}

export interface VideoInfo {
  duration: number;
  size_mb: number;
  width: number;
  height: number;
  fps: number;
  codec: string;
}

export interface JobStatus {
  job_id: string;
  filename: string;
  stage: string;
  progress: number;
  error: string | null;
  video_info: VideoInfo | null;
  candidates_count: number | null;
  logs: LogEntry[];
  updated_at: number;
}

export interface Candidate {
  candidate_id: number;
  start: number;
  end: number;
  text: string;
  score?: number;
  reasons?: string[];
}

export interface ClipResult {
  candidate_id: number;
  start: number;
  end: number;
  score: number;
  text: string;
  reasons: string[];
  clip_url: string;
  srt_url: string;
}

export interface JobResults {
  job_id: string;
  video_duration: number;
  total_candidates: number;
  clips: ClipResult[];
}

export interface TimelinePoint {
  time: number;
  score: number;
}

export interface LabelStats {
  total: number;
  highlights: number;
  non_highlights: number;
}

export interface TrainResult {
  status: string;
  metrics: {
    roc_auc: number;
    precision_at_5: number;
    train_samples: number;
    val_samples: number;
    positive_rate: number;
  };
}

export interface ModelStatus {
  trained: boolean;
  metrics?: TrainResult['metrics'];
}

@Injectable({ providedIn: 'root' })
export class ApiService {
  // Use relative URLs — Angular proxy forwards /api to backend
  private readonly base = '/api';

  constructor(private http: HttpClient) {
    console.log('[ApiService] initialized — using proxy at /api');
  }

  uploadVideo(file: File): Observable<{ job_id: string; status: string }> {
    const fd = new FormData();
    fd.append('file', file);
    return this.http.post<{ job_id: string; status: string }>(`${this.base}/jobs/upload`, fd);
  }

  getJobs(): Observable<JobStatus[]> {
    return this.http.get<JobStatus[]>(`${this.base}/jobs`);
  }

  getJobStatus(jobId: string): Observable<JobStatus> {
    return this.http.get<JobStatus>(`${this.base}/jobs/${jobId}/status`);
  }

  getJobResults(jobId: string): Observable<JobResults> {
    return this.http.get<JobResults>(`${this.base}/jobs/${jobId}/results`);
  }

  getTimeline(jobId: string): Observable<TimelinePoint[]> {
    return this.http.get<TimelinePoint[]>(`${this.base}/jobs/${jobId}/timeline`);
  }

  getCandidates(jobId: string): Observable<Candidate[]> {
    return this.http.get<Candidate[]>(`${this.base}/jobs/${jobId}/candidates`);
  }

  getTranscript(jobId: string): Observable<any[]> {
    return this.http.get<any[]>(`${this.base}/jobs/${jobId}/transcript`);
  }

  saveLabel(videoId: string, start: number, end: number, label: number): Observable<any> {
    return this.http.post(`${this.base}/labels`, { video_id: videoId, start, end, label });
  }

  getLabelStats(): Observable<LabelStats> {
    return this.http.get<LabelStats>(`${this.base}/labels/stats`);
  }

  trainModel(): Observable<TrainResult> {
    return this.http.post<TrainResult>(`${this.base}/train`, {});
  }

  rescoreJob(jobId: string): Observable<JobResults> {
    return this.http.post<JobResults>(`${this.base}/jobs/${jobId}/rescore`, {});
  }

  getModelStatus(): Observable<ModelStatus> {
    return this.http.get<ModelStatus>(`${this.base}/model/status`);
  }

  getHealth(): Observable<any> {
    return this.http.get(`${this.base}/health`);
  }

  clipUrl(path: string): string {
    return path;
  }
}
