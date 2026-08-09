from llm_security.datasets import RouterSample
from llm_security.models import ExpertFamily, ProjectCase
from llm_security.static_analysis import LightweightStaticAnalyzer


def router_samples() -> tuple[list[RouterSample], list[RouterSample]]:
    templates = {
        ExpertFamily.MEMORY_BOUNDS: (
            "void memory_%d(char *dst, const char *src, int len) "
            "{ memcpy(dst, src, len); }"
        ),
        ExpertFamily.LIFETIME_RESOURCE: (
            "void lifetime_%d(char *ptr) { free(ptr); ptr[0] = 1; }"
        ),
        ExpertFamily.INTEGER_SIZE_TYPE: (
            "void *integer_%d(int count) { return malloc(count * 16); }"
        ),
        ExpertFamily.TAINT_API_CONTRACT: (
            "void taint_%d(int fd) { char b[8]; read(fd,b,8); system(b); }"
        ),
        ExpertFamily.CONTROL_STATE_ERROR: (
            "int control_%d(char *p) { int state=0; parse(p); return state; }"
        ),
        ExpertFamily.CONCURRENCY_TOCTOU: (
            "int race_%d(char *p) { if(access(p,0)==0) return open(p,0); return -1; }"
        ),
    }
    analyzer = LightweightStaticAnalyzer(max_candidates=10)
    train: list[RouterSample] = []
    test: list[RouterSample] = []
    for family, template in templates.items():
        for index in range(4):
            case = ProjectCase(
                case_id=f"unit-{family.value}-{index}",
                project_id="unit-test",
                source_files={f"sample_{index}.c": template % index},
            )
            candidate = analyzer.analyze(case)[0]
            (train if index < 3 else test).append(RouterSample(candidate, [family]))
    return train, test
