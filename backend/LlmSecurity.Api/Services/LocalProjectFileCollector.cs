namespace LlmSecurity.Api.Services;

public sealed class LocalProjectFileCollector
{
    private static readonly HashSet<string> IgnoredDirectories =
        new(StringComparer.OrdinalIgnoreCase)
        {
            ".git",
            ".venv",
            "node_modules",
            "bin",
            "obj",
            "build",
            "dist",
            "vendor",
            ".web-data"
        };

    private static readonly HashSet<string> AllowedExtensions =
        new(StringComparer.OrdinalIgnoreCase)
        {
            ".c", ".cc", ".cpp", ".cxx",
            ".h", ".hh", ".hpp",
            ".cmake", ".txt", ".md",
            ".json", ".yaml", ".yml", ".toml"
        };

    private static readonly HashSet<string> AllowedFileNames =
        new(StringComparer.OrdinalIgnoreCase)
        {
            "CMakeLists.txt",
            "Makefile",
            "makefile",
            "meson.build"
        };

    public LocalProjectSnapshot Collect(
        string projectPath,
        string allowedRoot,
        int maxFiles,
        long maxTotalBytes,
        long maxSingleFileBytes)
    {
        if (string.IsNullOrWhiteSpace(projectPath))
            throw new ArgumentException("프로젝트 폴더 경로를 입력해주세요.", nameof(projectPath));

        var root = Path.GetFullPath(projectPath.Trim());
        var allowed = Path.GetFullPath(allowedRoot);

        if (!Directory.Exists(root))
            throw new DirectoryNotFoundException($"프로젝트 폴더를 찾을 수 없습니다: {root}");

        EnsureUnderAllowedRoot(root, allowed);

        var projectName = new DirectoryInfo(root).Name;
        if (string.IsNullOrWhiteSpace(projectName))
            projectName = "project";

        var files = new List<LocalProjectFile>();
        long totalBytes = 0;

        var pending = new Stack<string>();
        pending.Push(root);

        while (pending.Count > 0)
        {
            var current = pending.Pop();

            foreach (var directory in Directory.EnumerateDirectories(current))
            {
                var info = new DirectoryInfo(directory);

                // Junction/symlink를 따라가며 허용 루트 밖 파일을 읽는 것을 막는다.
                if ((info.Attributes & FileAttributes.ReparsePoint) != 0)
                    continue;

                if (IgnoredDirectories.Contains(info.Name))
                    continue;

                pending.Push(directory);
            }

            foreach (var filePath in Directory.EnumerateFiles(current))
            {
                var info = new FileInfo(filePath);

                if ((info.Attributes & FileAttributes.ReparsePoint) != 0)
                    continue;

                if (!ShouldInclude(info))
                    continue;

                if (info.Length > maxSingleFileBytes)
                {
                    throw new InvalidOperationException(
                        $"파일이 개발용 단일 파일 제한을 초과했습니다: {info.FullName}");
                }

                totalBytes += info.Length;
                if (totalBytes > maxTotalBytes)
                {
                    throw new InvalidOperationException(
                        "프로젝트가 개발용 총 업로드 용량 제한을 초과했습니다.");
                }

                if (files.Count >= maxFiles)
                {
                    throw new InvalidOperationException(
                        $"프로젝트 파일 수가 개발용 제한({maxFiles}개)을 초과했습니다.");
                }

                var relative = Path.GetRelativePath(root, info.FullName)
                    .Replace('\\', '/');

                // 브라우저의 webkitRelativePath와 같은 모양을 유지한다.
                var runtimeRelative = $"{projectName}/{relative}";

                files.Add(new LocalProjectFile(
                    info.FullName,
                    runtimeRelative,
                    info.Name,
                    info.Length));
            }
        }

        if (files.Count == 0)
        {
            throw new InvalidOperationException(
                "분석 가능한 C/C++ 소스 또는 프로젝트 파일을 찾지 못했습니다.");
        }

        if (!files.Any(file =>
                IsSourceExtension(Path.GetExtension(file.FileName))))
        {
            throw new InvalidOperationException(
                "C/C++ 소스 파일(.c/.cc/.cpp/.cxx/.h/.hh/.hpp)이 없습니다.");
        }

        return new LocalProjectSnapshot(
            root,
            projectName,
            files,
            totalBytes);
    }

    public static string ResolveConfiguredRoot(
        string contentRootPath,
        string? configuredRoot)
    {
        var value = string.IsNullOrWhiteSpace(configuredRoot)
            ? Path.Combine("..", "..")
            : configuredRoot.Trim();

        return Path.GetFullPath(
            Path.IsPathRooted(value)
                ? value
                : Path.Combine(contentRootPath, value));
    }

    private static bool ShouldInclude(FileInfo info)
    {
        if (AllowedFileNames.Contains(info.Name))
            return true;

        return AllowedExtensions.Contains(info.Extension);
    }

    private static bool IsSourceExtension(string extension) =>
        extension.Equals(".c", StringComparison.OrdinalIgnoreCase) ||
        extension.Equals(".cc", StringComparison.OrdinalIgnoreCase) ||
        extension.Equals(".cpp", StringComparison.OrdinalIgnoreCase) ||
        extension.Equals(".cxx", StringComparison.OrdinalIgnoreCase) ||
        extension.Equals(".h", StringComparison.OrdinalIgnoreCase) ||
        extension.Equals(".hh", StringComparison.OrdinalIgnoreCase) ||
        extension.Equals(".hpp", StringComparison.OrdinalIgnoreCase);

    private static void EnsureUnderAllowedRoot(string path, string allowedRoot)
    {
        var comparison = OperatingSystem.IsWindows()
            ? StringComparison.OrdinalIgnoreCase
            : StringComparison.Ordinal;

        var relative = Path.GetRelativePath(allowedRoot, path);

        if (relative == ".")
            return;

        if (Path.IsPathRooted(relative) ||
            relative.Equals("..", comparison) ||
            relative.StartsWith($"..{Path.DirectorySeparatorChar}", comparison) ||
            relative.StartsWith($"..{Path.AltDirectorySeparatorChar}", comparison))
        {
            throw new UnauthorizedAccessException(
                $"허용된 개발 경로 밖의 폴더입니다. 허용 루트: {allowedRoot}");
        }
    }
}

public sealed record LocalProjectFile(
    string FullPath,
    string RelativePath,
    string FileName,
    long Length);

public sealed record LocalProjectSnapshot(
    string RootPath,
    string ProjectName,
    IReadOnlyList<LocalProjectFile> Files,
    long TotalBytes);
