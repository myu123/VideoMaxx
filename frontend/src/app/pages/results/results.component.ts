import { Component, OnInit, ElementRef, ViewChild, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { ApiService, JobResults, ClipResult, TimelinePoint } from '../../services/api.service';

@Component({
  selector: 'app-results',
  imports: [CommonModule, RouterLink],
  templateUrl: './results.component.html',
  styleUrl: './results.component.scss',
})
export class ResultsComponent implements OnInit {
  jobId = '';
  results: JobResults | null = null;
  timeline: TimelinePoint[] = [];
  loading = true;
  error = '';

  @ViewChild('timelineCanvas') canvasRef!: ElementRef<HTMLCanvasElement>;

  constructor(private route: ActivatedRoute, private api: ApiService) {}

  ngOnInit() {
    this.jobId = this.route.snapshot.paramMap.get('jobId')!;
    this.api.getJobResults(this.jobId).subscribe({
      next: (r) => {
        this.results = r;
        this.loading = false;
        this.loadTimeline();
      },
      error: (e) => {
        this.error = 'Results not ready yet.';
        this.loading = false;
      },
    });
  }

  loadTimeline() {
    this.api.getTimeline(this.jobId).subscribe({
      next: (tl) => {
        this.timeline = tl;
        setTimeout(() => this.drawTimeline(), 100);
      },
    });
  }

  clipUrl(path: string): string {
    return this.api.clipUrl(path);
  }

  scoreClass(score: number): string {
    if (score >= 66) return 'high';
    if (score >= 33) return 'medium';
    return 'low';
  }

  formatTime(seconds: number): string {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  }

  drawTimeline() {
    const canvas = this.canvasRef?.nativeElement;
    if (!canvas || !this.timeline.length) return;

    const ctx = canvas.getContext('2d')!;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const pad = { top: 20, right: 20, bottom: 35, left: 45 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    const maxTime = Math.max(...this.timeline.map(p => p.time));
    const maxScore = Math.max(...this.timeline.map(p => p.score), 1);

    // Background
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = '#2a2a3e';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
    }

    // Gradient fill
    const gradient = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
    gradient.addColorStop(0, 'rgba(110, 231, 183, 0.3)');
    gradient.addColorStop(1, 'rgba(110, 231, 183, 0.0)');

    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top + plotH);
    this.timeline.forEach((p, i) => {
      const x = pad.left + (p.time / maxTime) * plotW;
      const y = pad.top + plotH - (p.score / maxScore) * plotH;
      if (i === 0) ctx.lineTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Line
    ctx.beginPath();
    ctx.strokeStyle = '#6ee7b7';
    ctx.lineWidth = 2;
    this.timeline.forEach((p, i) => {
      const x = pad.left + (p.time / maxTime) * plotW;
      const y = pad.top + plotH - (p.score / maxScore) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Highlight clip regions
    if (this.results) {
      ctx.fillStyle = 'rgba(59, 130, 246, 0.15)';
      for (const clip of this.results.clips) {
        const x1 = pad.left + (clip.start / maxTime) * plotW;
        const x2 = pad.left + (clip.end / maxTime) * plotW;
        ctx.fillRect(x1, pad.top, x2 - x1, plotH);
      }
    }

    // Axes labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'center';

    // X axis (time)
    const timeSteps = 6;
    for (let i = 0; i <= timeSteps; i++) {
      const t = (maxTime / timeSteps) * i;
      const x = pad.left + (t / maxTime) * plotW;
      ctx.fillText(this.formatTime(t), x, h - 8);
    }

    // Y axis (score)
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const score = (maxScore / 4) * (4 - i);
      const y = pad.top + (plotH / 4) * i;
      ctx.fillText(score.toFixed(0), pad.left - 8, y + 4);
    }

    // Axis title
    ctx.save();
    ctx.translate(12, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Score', 0, 0);
    ctx.restore();
  }
}
