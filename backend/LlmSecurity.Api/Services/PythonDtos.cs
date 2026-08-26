using System.Text.Json.Serialization;

namespace LlmSecurity.Api.Services;

public class PythonJobDto
{
    [JsonPropertyName("job_id")]
    public string JobId { get; set; } = "";

    [JsonPropertyName("project_name")]
    public string ProjectName { get; set; } = "";

    [JsonPropertyName("status")]
    public string Status { get; set; } = "";

    [JsonPropertyName("progress")]
    public int Progress { get; set; }

    [JsonPropertyName("message")]
    public string Message { get; set; } = "";

    [JsonPropertyName("file_count")]
    public int FileCount { get; set; }

    [JsonPropertyName("source_file_count")]
    public int SourceFileCount { get; set; }

    [JsonPropertyName("finding_count")]
    public int FindingCount { get; set; }

    [JsonPropertyName("validated_finding_count")]
    public int ValidatedFindingCount { get; set; }

    [JsonPropertyName("total_cost")]
    public double TotalCost { get; set; }

    [JsonPropertyName("error")]
    public string? Error { get; set; }
}
