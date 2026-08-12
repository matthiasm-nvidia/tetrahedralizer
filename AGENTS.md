# Reconstruction Sandbox

A sandbox to test various reconstruction methods to create a triangle mesh from a point cloud.
The point clouds are very noisy.

### Questions vs implementation

- When the user asks a **question** (explanation, review, "where is…", "how does…", "is it the case that…", etc.), **do not write or edit code**. Read and search the codebase as needed, then answer in text only.
- Only implement code changes when the user explicitly asks for a fix, feature, refactor, or similar implementation work.

## Coding style

- Match existing patterns in the module you are editing. Read surrounding code before writing new code.
- Always try to implement a feature in a simple way. I like simplicity
- Do only implement what is asked for. 
- Use the macros from CudaUtils.h and the DeviceBuffer object when possible when writing cuda code 

## Tests

Use `include/third_party/minunit.h`. Add tests when they cover real logic; skip trivial or obvious checks.