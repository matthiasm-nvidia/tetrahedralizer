#pragma once

#include <cstddef>

namespace tetrahedralizer
{

using GlUint = unsigned int;
using GlSize = std::ptrdiff_t;

bool glLoad();

void glGenBuffers(int n, GlUint* buffers);
void glBindBuffer(unsigned int target, GlUint buffer);
void glBufferData(unsigned int target, GlSize size, const void* data, unsigned int usage);
void glDeleteBuffers(int n, const GlUint* buffers);

GlUint glCreateShader(unsigned int type);
void glShaderSource(GlUint shader, int count, const char* const* string, const int* length);
void glCompileShader(GlUint shader);
void glDeleteShader(GlUint shader);

GlUint glCreateProgram();
void glAttachShader(GlUint program, GlUint shader);
void glLinkProgram(GlUint program);
void glGetProgramiv(GlUint program, unsigned int pname, int* params);
void glGetProgramInfoLog(GlUint program, int maxLength, int* length, char* infoLog);
void glDeleteProgram(GlUint program);
void glUseProgram(GlUint program);

int glGetUniformLocation(GlUint program, const char* name);
void glUniform1f(int location, float value);
void glUniform1i(int location, int value);
void glUniform3f(int location, float x, float y, float z);

void glDrawArrays(unsigned int mode, int first, int count);
void glDrawElements(unsigned int mode, int count, unsigned int type, const void* indices);

constexpr unsigned int kArrayBuffer = 0x8892;
constexpr unsigned int kElementArrayBuffer = 0x8893;
constexpr unsigned int kStaticDraw = 0x88E4;
constexpr unsigned int kDynamicDraw = 0x88E8;
constexpr unsigned int kFragmentShader = 0x8B30;
constexpr unsigned int kVertexShader = 0x8B31;
constexpr unsigned int kLinkStatus = 0x8B82;
constexpr unsigned int kPoints = 0x0000;
constexpr unsigned int kTriangles = 0x0004;
constexpr unsigned int kFloat = 0x1406;
constexpr unsigned int kUnsignedInt = 0x1405;

} // namespace tetrahedralizer
