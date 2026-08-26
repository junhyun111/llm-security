export type User = {
  id: string
  email: string
  displayName: string
}

export type AnalysisJob = {
  id: string
  projectName: string
  status: string
  progress: number
  message: string
  fileCount: number
  sourceFileCount: number
  findingCount: number
  validatedFindingCount: number
  totalCost: number
  errorMessage?: string | null
  createdAt: string
  updatedAt: string
  completedAt?: string | null
}

export type Dashboard = {
  totalScans: number
  completedScans: number
  totalFindings: number
  validatedFindings: number
  approvedPatches: number
  recentJobs: AnalysisJob[]
}

export type FindingBundle = {
  finding: {
    finding_id: string
    title: string
    file: string
    line_start: number
    line_end: number
    function: string
    root_cause: string
    consequence: string
    expert?: string
    supporting_experts?: string[]
    cwes?: string[]
    evidence_ids?: string[]
    evidence_against?: string[]
  }
  validation: {
    verdict: string
    confidence: number
    reasons?: string[]
  }
  candidate?: {
    cwe_hypotheses?: Array<{
      cwe: string
      confidence: number
      reasons?: string[]
    }>
  }
}

export type PatchBatch = {
  patch_id: string
  finding_ids: string[]
  status: string
  summary: string
  unified_diff: string
}

export type AnalysisPayload = {
  summary: {
    source_file_count: number
    candidate_count: number
    cwe_hypothesis_count: number
    finding_count: number
    validated_finding_count: number
    total_cost: number
    request_count: number
    submitted_expert_task_count: number
  }
  findings: FindingBundle[]
  patch_batch?: PatchBatch | null
}

export type AnalysisDetail = {
  job: AnalysisJob
  analysis?: AnalysisPayload | null
}
