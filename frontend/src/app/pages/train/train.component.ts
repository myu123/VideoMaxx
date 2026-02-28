import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { ApiService, LabelStats, ModelStatus, JobStatus } from '../../services/api.service';

@Component({
  selector: 'app-train',
  imports: [CommonModule, RouterLink],
  templateUrl: './train.component.html',
  styleUrl: './train.component.scss',
})
export class TrainComponent implements OnInit {
  labelStats: LabelStats = { total: 0, highlights: 0, non_highlights: 0 };
  modelStatus: ModelStatus = { trained: false };
  jobs: JobStatus[] = [];
  training = false;
  rescoring = false;
  trainResult: any = null;
  error = '';

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.api.getLabelStats().subscribe({ next: (s) => this.labelStats = s });
    this.api.getModelStatus().subscribe({ next: (s) => this.modelStatus = s });
    this.api.getJobs().subscribe({ next: (j) => this.jobs = j.filter(job => job.stage === 'complete') });
  }

  train() {
    this.training = true;
    this.error = '';
    this.trainResult = null;
    this.api.trainModel().subscribe({
      next: (r) => {
        this.training = false;
        this.trainResult = r;
        this.modelStatus = { trained: true, metrics: r.metrics };
      },
      error: (e) => {
        this.training = false;
        this.error = e.error?.detail || 'Training failed';
      },
    });
  }

  rescore(jobId: string) {
    this.rescoring = true;
    this.api.rescoreJob(jobId).subscribe({
      next: () => {
        this.rescoring = false;
        alert('Job re-scored! View updated results.');
      },
      error: (e) => {
        this.rescoring = false;
        this.error = e.error?.detail || 'Re-scoring failed';
      },
    });
  }
}
