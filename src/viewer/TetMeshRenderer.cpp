#include "TetMeshRenderer.h"

#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

#include <cstdio>

#ifndef GL_CURRENT_PROGRAM
#    define GL_CURRENT_PROGRAM 0x8B8D
#endif

namespace tetrahedralizer
{
namespace
{

constexpr int kTetEdgePairs[6][2] = {
    {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3},
};

constexpr int kTetFaceTriples[4][3] = {
    {0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3},
};

// The tet center travels with the face vertex so that the shrinking happens on the GPU.
struct FaceVertex
{
    Vec3 position;
    Vec3 center;
};

const char* kVertexShaderSource = R"(
uniform float uScale;
varying vec3 vPositionEye;

void main()
{
    vec3 center = gl_MultiTexCoord0.xyz;
    vec3 position = center + (gl_Vertex.xyz - center) * uScale;
    vPositionEye = vec3(gl_ModelViewMatrix * vec4(position, 1.0));
    gl_Position = gl_ModelViewProjectionMatrix * vec4(position, 1.0);
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
    vec3 baseColor = vec3(0.95, 0.75, 0.20);
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
        std::fprintf(stderr, "TetMeshRenderer: shader link failed:\n%s\n", log);
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

} // namespace

TetMeshRenderer::~TetMeshRenderer()
{
    clear();
}

void TetMeshRenderer::clear()
{
    if (m_vbo_edges != 0)
    {
        glDeleteBuffers(1, &m_vbo_edges);
        m_vbo_edges = 0;
    }
    if (m_vbo_faces != 0)
    {
        glDeleteBuffers(1, &m_vbo_faces);
        m_vbo_faces = 0;
    }
    if (m_shader_program != 0)
    {
        glDeleteProgram(m_shader_program);
        m_shader_program = 0;
    }
    m_num_edge_vertices = 0;
    m_num_face_vertices = 0;
    m_nodes.clear();
    m_tet_indices.clear();
}

void TetMeshRenderer::upload(const std::vector<Vec3>& nodes, const std::vector<int>& tet_indices)
{
    clear();

    m_nodes = nodes;
    m_tet_indices = tet_indices;
    buildBuffers();
}

void TetMeshRenderer::setClip(const Vec3& clip)
{
    if (clip == m_clip)
        return;

    m_clip = clip;
    buildBuffers();
}

void TetMeshRenderer::buildBuffers()
{
    m_num_edge_vertices = 0;
    m_num_face_vertices = 0;

    if (m_nodes.empty() || m_tet_indices.size() < 4)
        return;

    const int num_tets = static_cast<int>(m_tet_indices.size() / 4);
    std::vector<Vec3> segments;
    std::vector<FaceVertex> faces;
    segments.reserve(static_cast<std::size_t>(num_tets) * 12);
    faces.reserve(static_cast<std::size_t>(num_tets) * 12);

    for (int t = 0; t < num_tets; ++t)
    {
        const int base = t * 4;
        int corners[4];
        bool valid = true;
        for (int k = 0; k < 4; ++k)
        {
            corners[k] = m_tet_indices[static_cast<std::size_t>(base + k)];
            if (corners[k] < 0 || corners[k] >= static_cast<int>(m_nodes.size()))
            {
                valid = false;
                break;
            }
        }
        if (!valid)
            continue;

        Vec3 positions[4];
        for (int k = 0; k < 4; ++k)
            positions[k] = m_nodes[static_cast<std::size_t>(corners[k])];

        const Vec3 center = (positions[0] + positions[1] + positions[2] + positions[3]) * 0.25f;
        if (center.x > m_clip.x || center.y > m_clip.y || center.z > m_clip.z)
            continue;

        for (const auto& edge : kTetEdgePairs)
        {
            segments.push_back(positions[edge[0]]);
            segments.push_back(positions[edge[1]]);
        }

        for (const auto& face : kTetFaceTriples)
            for (const int corner : face)
                faces.push_back({positions[corner], center});
    }

    if (segments.empty())
        return;

    if (m_vbo_edges == 0)
        glGenBuffers(1, &m_vbo_edges);
    if (m_vbo_faces == 0)
        glGenBuffers(1, &m_vbo_faces);

    m_num_edge_vertices = static_cast<int>(segments.size());
    glBindBuffer(kArrayBuffer, m_vbo_edges);
    glBufferData(kArrayBuffer, static_cast<GlSize>(segments.size() * sizeof(Vec3)), segments.data(), kStaticDraw);

    m_num_face_vertices = static_cast<int>(faces.size());
    glBindBuffer(kArrayBuffer, m_vbo_faces);
    glBufferData(kArrayBuffer, static_cast<GlSize>(faces.size() * sizeof(FaceVertex)), faces.data(), kStaticDraw);
    glBindBuffer(kArrayBuffer, 0);
}

void TetMeshRenderer::ensureShaders() const
{
    if (m_shader_program == 0)
        m_shader_program = compileShaderProgram();
}

void TetMeshRenderer::render(bool wireframe, float scale) const
{
    if (wireframe)
    {
        if (m_num_edge_vertices <= 0 || m_vbo_edges == 0)
            return;

        glDisableClientState(GL_COLOR_ARRAY);
        glEnableClientState(GL_VERTEX_ARRAY);

        glBindBuffer(kArrayBuffer, m_vbo_edges);
        glVertexPointer(3, kFloat, 0, nullptr);

        glColor3f(0.95f, 0.75f, 0.20f);
        glDrawArrays(kLines, 0, m_num_edge_vertices);

        glDisableClientState(GL_VERTEX_ARRAY);
        glBindBuffer(kArrayBuffer, 0);
        return;
    }

    if (m_num_face_vertices <= 0 || m_vbo_faces == 0)
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
    glUniform1f(glGetUniformLocation(m_shader_program, "uScale"), scale);

    glDisableClientState(GL_COLOR_ARRAY);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    glBindBuffer(kArrayBuffer, m_vbo_faces);
    glVertexPointer(3, kFloat, sizeof(FaceVertex), nullptr);
    glTexCoordPointer(3, kFloat, sizeof(FaceVertex), reinterpret_cast<const void*>(sizeof(Vec3)));

    glDrawArrays(kTriangles, 0, m_num_face_vertices);

    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(kArrayBuffer, 0);
    glUseProgram(static_cast<GlUint>(previous_program));

    if (cull_face_was_enabled)
        glEnable(GL_CULL_FACE);
    if (lighting_was_enabled)
        glEnable(GL_LIGHTING);
}

} // namespace tetrahedralizer
