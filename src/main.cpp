#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "shader.hpp"
#include "particules.hpp"
#include <thread>
#include <chrono>
#include <atomic>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);

// settings
const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1280;

// camera
glm::vec3 cameraPos   = glm::vec3(0.0f, 0.0f, 90.0f);  
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp    = glm::vec3(0.0f, 1.0f, 0.0f);

bool firstMouse = true;
float yaw   = -90.0f;
float pitch =  0.0f;
float lastX =  800.0f / 2.0;
float lastY =  600.0 / 2.0;
float fov   =  45.0f;
bool pause = true;
// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

bool mouseDown = false;
glm::vec3 mouseWorld(0.0f);

glm::vec3 mouseToWorld(GLFWwindow* window,
                       const glm::mat4& view,
                       const glm::mat4& projection,
                       int fbW, int fbH,
                       float planeZ = 0.0f)
{
    double mx, my;
    glfwGetCursorPos(window, &mx, &my);

    float sx = 1.0f, sy = 1.0f;
    glfwGetWindowContentScale(window, &sx, &sy);
    mx *= sx;
    my *= sy;

    // Pixels framebuffer -> NDC [-1,1]
    float x = (2.0f * (float)mx) / (float)fbW - 1.0f;
    float y = 1.0f - (2.0f * (float)my) / (float)fbH;

    // NDC -> clip
    glm::vec4 rayClip(x, y, -1.0f, 1.0f);

    // clip -> eye
    glm::vec4 rayEye = glm::inverse(projection) * rayClip;
    rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0f, 0.0f);

    // eye -> world direction
    glm::mat4 invView = glm::inverse(view);
    glm::vec3 rayDirWorld = glm::normalize(glm::vec3(invView * rayEye));

    // origine du rayon = position caméra extraite de invView (plus fiable que cameraPos global)
    glm::vec3 rayOriginWorld = glm::vec3(invView[3]);

    // Intersection avec plan z = planeZ
    float denom = rayDirWorld.z;
    if (std::abs(denom) < 1e-6f)
        return rayOriginWorld + rayDirWorld * 100.0f;

    float t = (planeZ - rayOriginWorld.z) / denom;
    return rayOriginWorld + rayDirWorld * t;
}

int main() {
    if (!glfwInit()) {
        std::cerr << "GLFW init failed\n";
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "", nullptr, nullptr);
    glfwSwapInterval(1);
    if (!window) {
        std::cerr << "Window creation failed\n";
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to init GLAD\n";
        return 1;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;

    // Style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core"); // macOS OK en 3.3 core

    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << "\n";
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << "\n";

    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    Particules particule;

    int nBlue  = 50;
    int nGreen = 50;
    int nRed   = 50;

    glm::vec3 foyerPos = glm::vec3(0.f, 0.f, 0.f);

    glm::vec3 foyer1( foyerPos.x, foyerPos.y, foyerPos.z);
    glm::vec3 foyer2( 0.0f, 0.0f, 0.f);
    glm::vec3 foyer3(-foyerPos.x, -foyerPos.y, foyerPos.z);

    // création initiale
    particule.createParticules(foyer1, nBlue, foyer2, nGreen, foyer3, nRed);
    //particule.createSaturnSystem(glm::vec3(0,0,0), 1000);
    particule.renderInit();
    particule.initHeatmaps(256, 256);


    while (!glfwWindowShouldClose(window)) {

        float currentFrame = (float)glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        if (!pause)
            particule.simulate(deltaTime, mouseDown, mouseWorld);

        processInput(window);

        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.1f, 0.12f, 0.18f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Matrices
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 projection = glm::perspective(glm::radians(fov), (float)w / (float)h, 0.1f, 10000.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

        // état clic souris
        mouseDown = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);

        if (mouseDown) {
            mouseWorld = mouseToWorld(window, view, projection, w, h, 0.0f); // plan z=0
        }
        glDisable(GL_DEPTH_TEST);

        particule.trailShader.use();
        particule.trailShader.setMat4("model", model);
        particule.trailShader.setMat4("view", view);
        particule.trailShader.setMat4("projection", projection);
        particule.renderTrails();

        glEnable(GL_DEPTH_TEST);

        particule.shader.use();
        particule.shader.setMat4("model", model);
        particule.shader.setMat4("view", view);
        particule.shader.setMat4("projection", projection);

        particule.shader.setFloat("uViewportH", (float)h);
        particule.shader.setFloat("uFovY", glm::radians(fov));
        particule.shader.setVec3("uCameraPos", cameraPos);

        particule.renderUpdate();

        static float hmTimer = 0.0f;
        hmTimer += deltaTime;
        if (hmTimer > 0.1f) {
            particule.updateHeatmapsXY();
            hmTimer = 0.0f;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // --- UI ---
        ImGui::Begin("Controls");

        ImGui::Text("FPS: %f", 1.0f / (deltaTime > 0.0f ? deltaTime : 1.0f));
        ImGui::Checkbox("Pause", &pause);
        ImGui::Checkbox("Collision", &particule.collision);
        ImGui::Checkbox("Dislocation", &particule.enableDislocation);
        ImGui::Checkbox("Souris attracteur", &particule.mouseEffect);

        ImGui::Separator();

        static float pointScale = 10.0f;
        bool BradiusParticule = false;
        BradiusParticule |= ImGui::SliderFloat("Point scale", &particule.radiusParticule, 0.01f, 0.1f);

        if(BradiusParticule)
        {
            particule.radiusUpdate();
            particule.upload();
        }



        ImGui::SliderFloat3("Camera Pos", (float*)&cameraPos, 0.0f, 150.0f);

        ImGui::SliderFloat("Force de la souris", &particule.mouseStrength, 20.f, 100.0f);

        ImGui::SliderFloat("Taille du monde:", (float*)&particule.worldSize, 0.0f, 500.0f);

        ImGui::Separator();

        ImGui::Text("Masses (bleu, vert, rouge):");

        for (int i = 0; i < 3; ++i) {
                ImGui::PushID(i);
                ImGui::SetNextItemWidth(100);
                ImGui::DragFloat("##k", &particule.M[i], 1e7f, 1e4f, 1e10f);
                ImGui::SameLine();
                ImGui::PopID();
        }

        ImGui::Spacing();


        ImGui::Separator();

        bool BParticule = false;

        ImGui::SliderFloat("v0", &particule.v0, 0.0f, 3.0f);
        BParticule |= ImGui::SliderFloat3("Foyer Pos", (float*)&foyerPos, 0.0f, 40.0f);

        ImGui::Separator();
        ImGui::Text("Particles count");

        BParticule |= ImGui::SliderInt("Blue",  &nBlue,  0, 2000);
        BParticule |= ImGui::SliderInt("Green", &nGreen, 0, 2000);
        BParticule |= ImGui::SliderInt("Red",   &nRed,   0, 2000);

        if (BParticule)
        {
            foyer1.x = foyerPos.x; foyer1.y = foyerPos.y; foyer1.z = foyerPos.z;
            foyer3.x = -foyerPos.x; foyer2.y = -foyerPos.y; foyer2.z = foyerPos.z;
            particule.createParticules(foyer1, nBlue, foyer2, nGreen, foyer3, nRed);
            particule.upload();
        }

        ImGui::SliderInt("Trail length", &particule.trailMax, 2, 200);
        ImGui::SliderFloat("Trail alpha", &particule.trailAlpha, 0.0f, 1.0f);
        ImGui::SliderFloat("Trail width", &particule.trailWidth, 1.0f, 5.0f);

        ImGui::Separator();

        ImGui::SliderFloat("Break strength", &particule.breakStrength, 0.01f, 2.0f);
        ImGui::SliderFloat("Frag energy share", &particule.fragEnergyShare, 0.0f, 1.0f);
        ImGui::SliderInt("Max frags/collision", &particule.maxFragmentsPerCollision, 2, 15);

        ImGui::End();

        // --- ENERGY ---
        ImGui::Begin("Energie");
        ImGui::Text("Energie total: %f", particule.energyTotal);
        ImGui::Separator();
        ImGui::SliderFloat("Zoom", &particule.zoom, 0.5f, 15.0f);

        ImGui::Text("Colormap XY (worldSize=%.1f)", particule.worldSize);

        float z = particule.zoom;
        if (z < 1.0f) z = 1.0f;

        float u0 = 0.5f - 0.5f / z;
        float v0 = 0.5f - 0.5f / z;
        float u1 = 0.5f + 0.5f / z;
        float v1 = 0.5f + 0.5f / z;

        ImGui::Image((ImTextureID)(intptr_t)particule.texEnergy,
                     ImVec2(256, 256),
                     ImVec2(u0, v1), ImVec2(u1, v0));
        ImGui::End();

        // --- MASS ---
        ImGui::Begin("Masse");
        ImGui::Text("Masse total: %f", particule.massTotal);
        ImGui::Separator();
        ImGui::SliderFloat("Zoom", &particule.zoom, 0.5f, 15.0f);

        ImGui::Text("Colormap XY (worldSize=%.1f)", particule.worldSize);
        ImGui::Image((ImTextureID)(intptr_t)particule.texMass,
                     ImVec2(256, 256),
                     ImVec2(u0, v1), ImVec2(u1, v0));
        ImGui::End();

        // Rendering
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();


    }


    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    float cameraSpeed = 10.0f * deltaTime;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraFront;

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraFront;

    glm::vec3 right = glm::normalize(glm::cross(cameraFront, cameraUp));

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= right * cameraSpeed;

    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += right * cameraSpeed;

    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        cameraPos -= cameraUp * cameraSpeed;

    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        cameraPos += cameraUp * cameraSpeed;

    static bool spacePressedLastFrame = false;
    bool spacePressedNow = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;

    if (spacePressedNow && !spacePressedLastFrame)
        pause = !pause;

    spacePressedLastFrame = spacePressedNow;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}




