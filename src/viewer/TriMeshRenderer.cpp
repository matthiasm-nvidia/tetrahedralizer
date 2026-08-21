#include "TriMeshRenderer.h"

#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

#include <cmath>
#include <cstdio>
#include <limits>

#ifndef GL_CURRENT_PROGRAM
#    define GL_CURRENT_PROGRAM 0x8B8D
#endif

namespace tetrahedralizer
{
namespace
{

const char* kVertexShaderSource = R"(
varying vec3 vPositionEye;
varying float vSize;

void main()
{
    vec4 positionEye = gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0);
    vPositionEye = positionEye.xyz;
    vSize = gl_MultiTexCoord0.x;
    gl_ClipVertex = positionEye;
    gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);
}
)";

const char* kFragmentShaderSource = R"(
varying vec3 vPositionEye;
varying float vSize;
uniform int uColorize;
uniform float uLogMin;
uniform float uLogMax;

vec3 sizeColor(float t)
{
    t = clamp(t, 0.0, 1.0);
    if (t < 0.25)
        return mix(vec3(0.15, 0.25, 0.85), vec3(0.10, 0.70, 0.90), t / 0.25);
    if (t < 0.50)
        return mix(vec3(0.10, 0.70, 0.90), vec3(0.20, 0.80, 0.30), (t - 0.25) / 0.25);
    if (t < 0.75)
        return mix(vec3(0.20, 0.80, 0.30), vec3(0.95, 0.85, 0.20), (t - 0.50) / 0.25);
    return mix(vec3(0.95, 0.85, 0.20), vec3(0.85, 0.15, 0.10), (t - 0.75) / 0.25);
}

void main()
{
    vec3 normal = normalize(cross(dFdx(vPositionEye), dFdy(vPositionEye)));
    if (dot(normal, normal) < 1.0e-12)
        normal = vec3(0.0, 0.0, 1.0);

    vec3 lightDirection = normalize(vec3(0.3, 0.6, 1.0));
    float diffuse = abs(dot(normal, lightDirection));

    vec3 baseColor = vec3(0.72, 0.76, 0.82);
    if (uColorize != 0)
    {
        float logSize = log(max(vSize, 1.0e-12));
        float range = max(uLogMax - uLogMin, 1.0e-12);
        float t = clamp((uLogMax - logSize) / range, 0.0, 1.0);
        baseColor = sizeColor(t);
    }

    gl_FragColor = vec4(baseColor * (0.30 + 0.70 * diffuse), 1.0);
}
)";

GlUint compileShaderProgram()
{
    const GlUint vertex_shader = glCreateShader(kVertexShader);
    const GlUint fragment_shader = glCreateShader(kFragmentShader);
    glShaderSource(vertex_shader, 1, &kVertexShaderSource, nullptr);
    glShaderSource(fragment_shader, 1, &kFragmentShaderSource, nullptr);
    glCompileShader(vertex_shader);
    glCompileShader(fragment_shader);

    const GlUint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    int linked = GL_FALSE;
    glGetProgramiv(program, kLinkStatus, &linked);
    if (!linked)
    {
        char log[512] = {};
        glGetProgramInfoLog(program, static_cast<int>(sizeof(log)), nullptr, log);
        std::fprintf(stderr, "TriMeshRenderer: shader link failed:\n%s\n", log);
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

} // namespace

TriMeshRenderer::~TriMeshRenderer()
{
    clear();
}

void TriMeshRenderer::clearSizeField()
{
    if (m_vbo_sizes != 0)
    {
        glDeleteBuffers(1, &m_vbo_sizes);
        m_vbo_sizes = 0;
    }
    m_colorize = false;
}

void TriMeshRenderer::clear()
{
    if (m_vbo_vertices != 0)
    {
        glDeleteBuffers(1, &m_vbo_vertices);
        m_vbo_vertices = 0;
    }
    if (m_ibo_indices != 0)
    {
        glDeleteBuffers(1, &m_ibo_indices);
        m_ibo_indices = 0;
    }
    if (m_shader_program != 0)
    {
        glDeleteProgram(m_shader_program);
        m_shader_program = 0;
    }
    clearSizeField();
    m_num_indices = 0;
    m_num_vertices = 0;
}

void TriMeshRenderer::upload(const std::vector<Vec3>& positions,
                             const std::vector<std::uint32_t>& triangle_indices)
{
    if (positions.empty() || triangle_indices.empty() ||
        triangle_indices.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        clear();
        return;
    }

    m_num_indices = static_cast<int>(triangle_indices.size());
    m_num_vertices = static_cast<int>(positions.size());
    clearSizeField();
    if (m_vbo_vertices == 0)
        glGenBuffers(1, &m_vbo_vertices);
    if (m_ibo_indices == 0)
        glGenBuffers(1, &m_ibo_indices);

    glBindBuffer(kArrayBuffer, m_vbo_vertices);
    glBufferData(kArrayBuffer, static_cast<GlSize>(positions.size() * sizeof(Vec3)), positions.data(), kStaticDraw);
    glBindBuffer(kElementArrayBuffer, m_ibo_indices);
    glBufferData(kElementArrayBuffer,
                 static_cast<GlSize>(triangle_indices.size() * sizeof(std::uint32_t)),
                 triangle_indices.data(), kStaticDraw);
    glBindBuffer(kArrayBuffer, 0);
    glBindBuffer(kElementArrayBuffer, 0);
}

void TriMeshRenderer::uploadSizeField(const std::vector<float>& sizes)
{
    if (m_num_vertices <= 0 || sizes.size() != static_cast<std::size_t>(m_num_vertices))
    {
        clearSizeField();
        return;
    }

    if (m_vbo_sizes == 0)
        glGenBuffers(1, &m_vbo_sizes);

    glBindBuffer(kArrayBuffer, m_vbo_sizes);
    glBufferData(kArrayBuffer, static_cast<GlSize>(sizes.size() * sizeof(float)), sizes.data(), kStaticDraw);
    glBindBuffer(kArrayBuffer, 0);
    m_colorize = true;
}

void TriMeshRenderer::setSizeFieldRange(float minSize, float maxSize)
{
    const float lo = minSize > 1.0e-12f && std::isfinite(minSize) ? minSize : 1.0e-12f;
    const float hi = maxSize > lo && std::isfinite(maxSize) ? maxSize : lo * 1.0001f;
    m_log_min = std::log(lo);
    m_log_max = std::log(hi);
}

void TriMeshRenderer::ensureShaders() const
{
    if (m_shader_program == 0)
        m_shader_program = compileShaderProgram();
}

void TriMeshRenderer::render(bool wireframe) const
{
    if (m_num_indices <= 0 || m_vbo_vertices == 0 || m_ibo_indices == 0)
        return;

    ensureShaders();
    if (m_shader_program == 0)
        return;

    const GLboolean lighting_was_enabled = glIsEnabled(GL_LIGHTING);
    const GLboolean cull_face_was_enabled = glIsEnabled(GL_CULL_FACE);
    int previous_program = 0;
    glGetIntegerv(GL_CURRENT_PROGRAM, &previous_program);

    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE);
    glUseProgram(m_shader_program);
    glUniform1i(glGetUniformLocation(m_shader_program, "uColorize"), m_colorize ? 1 : 0);
    glUniform1f(glGetUniformLocation(m_shader_program, "uLogMin"), m_log_min);
    glUniform1f(glGetUniformLocation(m_shader_program, "uLogMax"), m_log_max);

    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(kArrayBuffer, m_vbo_vertices);
    glVertexPointer(3, kFloat, 0, nullptr);
    if (m_colorize && m_vbo_sizes != 0)
    {
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);
        glBindBuffer(kArrayBuffer, m_vbo_sizes);
        glTexCoordPointer(1, kFloat, 0, nullptr);
    }
    glBindBuffer(kElementArrayBuffer, m_ibo_indices);

    if (wireframe)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDrawElements(kTriangles, m_num_indices, kUnsignedInt, nullptr);
    if (wireframe)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(kArrayBuffer, 0);
    glBindBuffer(kElementArrayBuffer, 0);
    glUseProgram(static_cast<GlUint>(previous_program));

    if (cull_face_was_enabled)
        glEnable(GL_CULL_FACE);
    if (lighting_was_enabled)
        glEnable(GL_LIGHTING);
}

} // namespace tetrahedralizer
