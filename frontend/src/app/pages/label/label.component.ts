import { Component, OnInit, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute } from '@angular/router';
import { ApiService, Candidate } from '../../services/api.service';

@Component({
  selector: 'app-label',
  imports: [CommonModule],
  templateUrl: './label.component.html',
  styleUrl: './label.component.scss',
})
export class LabelComponent implements OnInit {
  jobId = '';
  candidates: Candidate[] = [];
  currentIndex = 0;
  labeledCount = 0;
  saving = false;
  videoUrl = '';

  constructor(private route: ActivatedRoute, private api: ApiService) {}

  ngOnInit() {
    this.jobId = this.route.snapshot.paramMap.get('jobId')!;
    this.api.getCandidates(this.jobId).subscribe({
      next: (c) => {
        this.candidates = c;
      },
    });
  }

  get current(): Candidate | null {
    return this.candidates[this.currentIndex] ?? null;
  }

  get progress(): number {
    if (!this.candidates.length) return 0;
    return Math.round((this.currentIndex / this.candidates.length) * 100);
  }

  formatTime(seconds: number): string {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  }

  @HostListener('window:keydown', ['$event'])
  onKeyDown(e: KeyboardEvent) {
    if (e.key === '1') this.label(1);
    else if (e.key === '0') this.label(0);
    else if (e.key === 's' || e.key === 'S') this.skip();
  }

  label(value: number) {
    if (!this.current || this.saving) return;
    this.saving = true;
    this.api.saveLabel(this.jobId, this.current.start, this.current.end, value).subscribe({
      next: () => {
        this.labeledCount++;
        this.saving = false;
        this.next();
      },
      error: () => {
        this.saving = false;
      },
    });
  }

  skip() {
    this.next();
  }

  next() {
    if (this.currentIndex < this.candidates.length - 1) {
      this.currentIndex++;
    }
  }

  prev() {
    if (this.currentIndex > 0) {
      this.currentIndex--;
    }
  }
}
