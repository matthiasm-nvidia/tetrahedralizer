#include "GlCore.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace tetrahedralizer
{
namespace
{
using PFNGLGENBUFFERSPROC = void (*)(int, unsigned int*);
using PFNGLBINDBUFFERPROC = void (*)(unsigned int, unsigned int);
using PFNGLBUFFERDATAPROC = void (*)(unsigned int, GlSize, const void*, unsigned int);
using PFNGLDELETEBUFFERSPROC = void (*)(int, const unsigned int*);
using PFNGLCREATESHADERPROC = unsigned int (*)(unsigned int);
using PFNGLSHADERSOURCEPROC = void (*)(unsigned int, int, const char* const*, const int*);
using PFNGLCOMPILESHADERPROC = void (*)(unsigned int);
using PFNGLDELETESHADERPROC = void (*)(unsigned int);
using PFNGLCREATEPROGRAMPROC = unsigned int (*)();
using PFNGLATTACHSHADERPROC = void (*)(unsigned int, unsigned int);
using PFNGLLINKPROGRAMPROC = void (*)(unsigned int);
using PFNGLGETPROGRAMIVPROC = void (*)(unsigned int, unsigned int, int*);
using PFNGLGETPROGRAMINFOLOGPROC = void (*)(unsigned int, int, int*, char*);
using PFNGLDELETEPROGRAMPROC = void (*)(unsigned int);
using PFNGLUSEPROGRAMPROC = void (*)(unsigned int);
using PFNGLGETUNIFORMLOCATIONPROC = int (*)(unsigned int, const char*);
using PFNGLUNIFORM1FPROC = void (*)(int, float);
using PFNGLUNIFORM1IPROC = void (*)(int, int);
using PFNGLUNIFORM3FPROC = void (*)(int, float, float, float);
using PFNGLDRAWARRAYSPROC = void (*)(unsigned int, int, int);
using PFNGLDRAWELEMENTSPROC = void (*)(unsigned int, int, unsigned int, const void*);

PFNGLGENBUFFERSPROC p_glGenBuffers = nullptr;
PFNGLBINDBUFFERPROC p_glBindBuffer = nullptr;
PFNGLBUFFERDATAPROC p_glBufferData = nullptr;
PFNGLDELETEBUFFERSPROC p_glDeleteBuffers = nullptr;
PFNGLCREATESHADERPROC p_glCreateShader = nullptr;
PFNGLSHADERSOURCEPROC p_glShaderSource = nullptr;
PFNGLCOMPILESHADERPROC p_glCompileShader = nullptr;
PFNGLDELETESHADERPROC p_glDeleteShader = nullptr;
PFNGLCREATEPROGRAMPROC p_glCreateProgram = nullptr;
PFNGLATTACHSHADERPROC p_glAttachShader = nullptr;
PFNGLLINKPROGRAMPROC p_glLinkProgram = nullptr;
PFNGLGETPROGRAMIVPROC p_glGetProgramiv = nullptr;
PFNGLGETPROGRAMINFOLOGPROC p_glGetProgramInfoLog = nullptr;
PFNGLDELETEPROGRAMPROC p_glDeleteProgram = nullptr;
PFNGLUSEPROGRAMPROC p_glUseProgram = nullptr;
PFNGLGETUNIFORMLOCATIONPROC p_glGetUniformLocation = nullptr;
PFNGLUNIFORM1FPROC p_glUniform1f = nullptr;
PFNGLUNIFORM1IPROC p_glUniform1i = nullptr;
PFNGLUNIFORM3FPROC p_glUniform3f = nullptr;
PFNGLDRAWARRAYSPROC p_glDrawArrays = nullptr;
PFNGLDRAWELEMENTSPROC p_glDrawElements = nullptr;
} // namespace

bool glLoad()
{
    p_glGenBuffers = reinterpret_cast<PFNGLGENBUFFERSPROC>(glfwGetProcAddress("glGenBuffers"));
    p_glBindBuffer = reinterpret_cast<PFNGLBINDBUFFERPROC>(glfwGetProcAddress("glBindBuffer"));
    p_glBufferData = reinterpret_cast<PFNGLBUFFERDATAPROC>(glfwGetProcAddress("glBufferData"));
    p_glDeleteBuffers = reinterpret_cast<PFNGLDELETEBUFFERSPROC>(glfwGetProcAddress("glDeleteBuffers"));
    p_glCreateShader = reinterpret_cast<PFNGLCREATESHADERPROC>(glfwGetProcAddress("glCreateShader"));
    p_glShaderSource = reinterpret_cast<PFNGLSHADERSOURCEPROC>(glfwGetProcAddress("glShaderSource"));
    p_glCompileShader = reinterpret_cast<PFNGLCOMPILESHADERPROC>(glfwGetProcAddress("glCompileShader"));
    p_glDeleteShader = reinterpret_cast<PFNGLDELETESHADERPROC>(glfwGetProcAddress("glDeleteShader"));
    p_glCreateProgram = reinterpret_cast<PFNGLCREATEPROGRAMPROC>(glfwGetProcAddress("glCreateProgram"));
    p_glAttachShader = reinterpret_cast<PFNGLATTACHSHADERPROC>(glfwGetProcAddress("glAttachShader"));
    p_glLinkProgram = reinterpret_cast<PFNGLLINKPROGRAMPROC>(glfwGetProcAddress("glLinkProgram"));
    p_glGetProgramiv = reinterpret_cast<PFNGLGETPROGRAMIVPROC>(glfwGetProcAddress("glGetProgramiv"));
    p_glGetProgramInfoLog = reinterpret_cast<PFNGLGETPROGRAMINFOLOGPROC>(glfwGetProcAddress("glGetProgramInfoLog"));
    p_glDeleteProgram = reinterpret_cast<PFNGLDELETEPROGRAMPROC>(glfwGetProcAddress("glDeleteProgram"));
    p_glUseProgram = reinterpret_cast<PFNGLUSEPROGRAMPROC>(glfwGetProcAddress("glUseProgram"));
    p_glGetUniformLocation = reinterpret_cast<PFNGLGETUNIFORMLOCATIONPROC>(glfwGetProcAddress("glGetUniformLocation"));
    p_glUniform1f = reinterpret_cast<PFNGLUNIFORM1FPROC>(glfwGetProcAddress("glUniform1f"));
    p_glUniform1i = reinterpret_cast<PFNGLUNIFORM1IPROC>(glfwGetProcAddress("glUniform1i"));
    p_glUniform3f = reinterpret_cast<PFNGLUNIFORM3FPROC>(glfwGetProcAddress("glUniform3f"));
    p_glDrawArrays = reinterpret_cast<PFNGLDRAWARRAYSPROC>(glfwGetProcAddress("glDrawArrays"));
    p_glDrawElements = reinterpret_cast<PFNGLDRAWELEMENTSPROC>(glfwGetProcAddress("glDrawElements"));

    return p_glGenBuffers && p_glCreateProgram && p_glDrawArrays && p_glDrawElements && p_glUniform1f && p_glUniform1i &&
           p_glUniform3f;
}

void glGenBuffers(int n, GlUint* buffers)
{
    p_glGenBuffers(n, buffers);
}

void glBindBuffer(unsigned int target, GlUint buffer)
{
    p_glBindBuffer(target, buffer);
}

void glBufferData(unsigned int target, GlSize size, const void* data, unsigned int usage)
{
    p_glBufferData(target, size, data, usage);
}

void glDeleteBuffers(int n, const GlUint* buffers)
{
    p_glDeleteBuffers(n, buffers);
}

GlUint glCreateShader(unsigned int type)
{
    return p_glCreateShader(type);
}

void glShaderSource(GlUint shader, int count, const char* const* string, const int* length)
{
    p_glShaderSource(shader, count, string, length);
}

void glCompileShader(GlUint shader)
{
    p_glCompileShader(shader);
}

void glDeleteShader(GlUint shader)
{
    p_glDeleteShader(shader);
}

GlUint glCreateProgram()
{
    return p_glCreateProgram();
}

void glAttachShader(GlUint program, GlUint shader)
{
    p_glAttachShader(program, shader);
}

void glLinkProgram(GlUint program)
{
    p_glLinkProgram(program);
}

void glGetProgramiv(GlUint program, unsigned int pname, int* params)
{
    p_glGetProgramiv(program, pname, params);
}

void glGetProgramInfoLog(GlUint program, int maxLength, int* length, char* infoLog)
{
    p_glGetProgramInfoLog(program, maxLength, length, infoLog);
}

void glDeleteProgram(GlUint program)
{
    p_glDeleteProgram(program);
}

void glUseProgram(GlUint program)
{
    p_glUseProgram(program);
}

int glGetUniformLocation(GlUint program, const char* name)
{
    return p_glGetUniformLocation(program, name);
}

void glUniform1f(int location, float value)
{
    p_glUniform1f(location, value);
}

void glUniform1i(int location, int value)
{
    p_glUniform1i(location, value);
}

void glUniform3f(int location, float x, float y, float z)
{
    p_glUniform3f(location, x, y, z);
}

void glDrawArrays(unsigned int mode, int first, int count)
{
    p_glDrawArrays(mode, first, count);
}

void glDrawElements(unsigned int mode, int count, unsigned int type, const void* indices)
{
    p_glDrawElements(mode, count, type, indices);
}

} // namespace tetrahedralizer
