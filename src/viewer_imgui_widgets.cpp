#include "viewer_imgui_widgets.h"

#include "imgui.h"
#include "imgui_internal.h"
#include "viewer_imgui_theme.h"

namespace viewer::imgui_widgets
{
namespace
{
constexpr ImU32 k_field_bg = IM_COL32(22, 22, 22, 255);
constexpr ImU32 k_field_fill = IM_COL32(56, 56, 56, 255);
constexpr ImU32 k_border = IM_COL32(120, 120, 120, 255);
constexpr ImU32 k_text = IM_COL32(224, 224, 224, 255);
constexpr ImU32 k_checkbox_bg = IM_COL32(192, 192, 192, 255);
constexpr ImU32 k_checkbox_mark = IM_COL32(38, 38, 38, 255);
constexpr ImU32 k_slider_grab = IM_COL32(99, 99, 99, 255);
constexpr ImU32 k_slider_grab_active = IM_COL32(125, 125, 125, 255);
constexpr float k_checkbox_size = 18.0f;
constexpr float k_checkbox_radius = 4.0f;
constexpr float k_slider_height = 28.6f;
constexpr float k_corner_radius = 3.0f;

void draw_centered_checkmark(ImDrawList* draw_list, const ImRect& box, ImU32 color)
{
    const float size = ImMin(box.GetWidth(), box.GetHeight());
    const ImVec2 center = box.GetCenter();
    const ImVec2 p1(center.x - size * 0.17f, center.y + size * 0.02f);
    const ImVec2 p2(center.x - size * 0.03f, center.y + size * 0.20f);
    const ImVec2 p3(center.x + size * 0.21f, center.y - size * 0.17f);
    draw_list->PathLineTo(p1);
    draw_list->PathLineTo(p2);
    draw_list->PathLineTo(p3);
    draw_list->PathStroke(color, false, 2.0f);
}

void plain_label(const char* text)
{
    ImGui::PushStyleColor(ImGuiCol_Text, k_text);
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
}

void draw_slider_border()
{
    if (ImGui::IsItemVisible())
    {
        const ImVec2 min = ImGui::GetItemRectMin();
        const ImVec2 max = ImGui::GetItemRectMax();
        ImGui::GetWindowDrawList()->AddRect(min, max, k_border, k_corner_radius);
    }
}

void push_slider_style()
{
    ImGui::PushStyleColor(ImGuiCol_FrameBg, k_field_bg);
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, k_field_bg);
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, k_field_fill);
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, k_slider_grab);
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, k_slider_grab_active);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, k_corner_radius);
    ImGui::PushStyleVar(ImGuiStyleVar_GrabMinSize, 8.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f, (k_slider_height - ImGui::GetTextLineHeight()) * 0.5f));
}

void pop_slider_style()
{
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor(5);
}
} // namespace

void section_heading(const char* text)
{
    ImFont* bold_font = viewer::imgui_theme::ui_bold_font();
    if (bold_font != nullptr)
    {
        ImGui::PushFont(bold_font);
    }

    ImGui::PushStyleColor(ImGuiCol_Text, k_text);
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();

    if (bold_font != nullptr)
    {
        ImGui::PopFont();
    }
}

bool collapsing_section(const char* text, bool default_open)
{
    ImFont* bold_font = viewer::imgui_theme::ui_bold_font();
    if (bold_font != nullptr)
        ImGui::PushFont(bold_font);

    ImGui::PushStyleColor(ImGuiCol_Text, k_text);
    ImGui::PushStyleColor(ImGuiCol_Header, k_field_bg);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, k_field_fill);
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, k_field_fill);

    ImGuiTreeNodeFlags flags = 0;
    if (default_open)
        flags |= ImGuiTreeNodeFlags_DefaultOpen;
    const bool open = ImGui::CollapsingHeader(text, flags);

    ImGui::PopStyleColor(4);
    if (bold_font != nullptr)
        ImGui::PopFont();
    return open;
}

void section_separator()
{
    ImGui::PushStyleColor(ImGuiCol_Separator, k_border);
    ImGui::Separator();
    ImGui::PopStyleColor();
    ImGui::Spacing();
}

bool checkbox(const char* label, bool* value)
{
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    if (window->SkipItems)
    {
        return false;
    }

    ImGuiContext& g = *GImGui;
    const ImGuiStyle& style = g.Style;
    ImGui::PushID(label);
    const ImGuiID id = window->GetID("##cb");

    const ImVec2 label_size = ImGui::CalcTextSize(label);
    const ImVec2 pos = window->DC.CursorPos;
    const ImRect check_bb(pos, ImVec2(pos.x + k_checkbox_size, pos.y + k_checkbox_size));
    const float row_h = ImMax(k_checkbox_size, label_size.y);
    const ImRect total_bb(pos, ImVec2(pos.x + k_checkbox_size + style.ItemInnerSpacing.x + label_size.x, pos.y + row_h));

    ImGui::ItemSize(total_bb, style.FramePadding.y);
    if (!ImGui::ItemAdd(total_bb, id))
    {
        ImGui::PopID();
        return false;
    }

    bool hovered = false;
    bool held = false;
    const bool pressed = ImGui::ButtonBehavior(total_bb, id, &hovered, &held);
    if (pressed)
    {
        *value = !*value;
    }

    window->DrawList->AddRectFilled(check_bb.Min, check_bb.Max, k_checkbox_bg, k_checkbox_radius);
    if (*value)
    {
        draw_centered_checkmark(window->DrawList, check_bb, k_checkbox_mark);
    }

    const ImVec2 label_pos(check_bb.Max.x + style.ItemInnerSpacing.x, pos.y + (row_h - label_size.y) * 0.5f);
    ImGui::GetWindowDrawList()->AddText(label_pos, k_text, label);

    ImGui::PopID();
    return pressed;
}

bool radio_button(const char* label, int* value, int option)
{
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    if (window->SkipItems)
    {
        return false;
    }

    ImGuiContext& g = *GImGui;
    const ImGuiStyle& style = g.Style;
    ImGui::PushID(label);
    const ImGuiID id = window->GetID("##rb");

    const ImVec2 label_size = ImGui::CalcTextSize(label);
    const ImVec2 pos = window->DC.CursorPos;
    const ImRect check_bb(pos, ImVec2(pos.x + k_checkbox_size, pos.y + k_checkbox_size));
    const float row_h = ImMax(k_checkbox_size, label_size.y);
    const ImRect total_bb(pos, ImVec2(pos.x + k_checkbox_size + style.ItemInnerSpacing.x + label_size.x, pos.y + row_h));

    ImGui::ItemSize(total_bb, style.FramePadding.y);
    if (!ImGui::ItemAdd(total_bb, id))
    {
        ImGui::PopID();
        return false;
    }

    bool hovered = false;
    bool held = false;
    const bool pressed = ImGui::ButtonBehavior(total_bb, id, &hovered, &held);
    if (pressed)
    {
        *value = option;
    }

    const ImVec2 center = check_bb.GetCenter();
    const float radius = k_checkbox_size * 0.5f;
    window->DrawList->AddCircleFilled(center, radius, k_checkbox_bg);
    if (*value == option)
    {
        window->DrawList->AddCircleFilled(center, radius * 0.5f, k_checkbox_mark);
    }

    const ImVec2 label_pos(check_bb.Max.x + style.ItemInnerSpacing.x, pos.y + (row_h - label_size.y) * 0.5f);
    ImGui::GetWindowDrawList()->AddText(label_pos, k_text, label);

    ImGui::PopID();
    return pressed;
}

bool slider_float(const char* label, float* value, float min_value, float max_value, const char* display_format, float power)
{
    ImGui::PushID(label);
    plain_label(label);
    push_slider_style();

    ImGui::PushItemWidth(-1.0f);
    const bool changed = ImGui::SliderFloat("##slider", value, min_value, max_value, display_format, power);
    ImGui::PopItemWidth();

    draw_slider_border();
    pop_slider_style();
    ImGui::PopID();
    return changed;
}

bool slider_int(const char* label, int* value, int min_value, int max_value, const char* display_format)
{
    ImGui::PushID(label);
    push_slider_style();
    ImGui::AlignTextToFramePadding();
    plain_label(label);
    ImGui::SameLine();

    ImGui::PushItemWidth(-1.0f);
    const bool changed = ImGui::SliderInt("##slider", value, min_value, max_value, display_format);
    ImGui::PopItemWidth();

    draw_slider_border();
    pop_slider_style();
    ImGui::PopID();
    return changed;
}

bool button(const char* label, float width)
{
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, k_corner_radius);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 7.0f));
    ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(30, 100, 156, 255));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, IM_COL32(37, 121, 187, 255));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, IM_COL32(23, 78, 122, 255));

    ImFont* bold_font = viewer::imgui_theme::ui_bold_font();
    if (bold_font != nullptr)
    {
        ImGui::PushFont(bold_font);
    }

    const bool pressed = ImGui::Button(label, ImVec2(width, 0.0f));

    if (bold_font != nullptr)
    {
        ImGui::PopFont();
    }

    if (ImGui::IsItemVisible())
    {
        const ImVec2 min = ImGui::GetItemRectMin();
        const ImVec2 max = ImGui::GetItemRectMax();
        ImGui::GetWindowDrawList()->AddRect(min, max, k_border, k_corner_radius);
    }

    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(2);
    return pressed;
}

bool button_full_width(const char* label)
{
    return button(label, ImGui::GetContentRegionAvail().x);
}

} // namespace viewer::imgui_widgets
