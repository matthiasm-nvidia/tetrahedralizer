#include "TriMeshRenderer.h"

#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

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

void main()
{
    vec4 positionEye = gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0);
    vPositionEye = positionEye.xyz;
    gl_ClipVertex = positionEye;
    gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);
}
)";

const char* kFragmentShaderSource = R"(
varying vec3 vPositionEye;

void main()
{
    vec3 normal = normalize(cross(dFdx(vPositionEye), dFdy(vPositionEye)));
    if (dot(normal, normal) < 1.0e-12)
        normal = vec3(0.0, 0.0, 1.0);

    vec3 lightDirection = normalize(vec3(0.3, 0.6, 1.0));
    float diffuse = abs(dot(normal, lightDirection));
    vec3 baseColor = vec3(0.72, 0.76, 0.82);
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
    m_num_indices = 0;
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

    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(kArrayBuffer, m_vbo_vertices);
    glVertexPointer(3, kFloat, 0, nullptr);
    glBindBuffer(kElementArrayBuffer, m_ibo_indices);

    if (wireframe)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDrawElements(kTriangles, m_num_indices, kUnsignedInt, nullptr);
    if (wireframe)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

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
