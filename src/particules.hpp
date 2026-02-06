#pragma once 
#include <iostream>
#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "shader.hpp"
#include <random>
#include <cmath>
#include <cstddef>

struct Vertex
{
    float x, y, z;     // position
    float r, g, b;     // couleur (affichage)
    float vx, vy, vz;  // vitesse
    int   type;        // 0=bleu, 1=vert, 2=rouge
    float energy;
    float mass;
    float radius;
    int   alive;  
};

struct TrailGPU {
    float x,y,z;
    float r,g,b,a;
};

class Particules {

public:
	unsigned int VBO, VAO;
	std::vector<Vertex> vertices;
	const char* shader_vs = "shader/shader.vs";
	const char* shader_fs = "shader/shader.fs";
	Shader shader;

	const char* trail_vs = "shader/trail.vs";
	const char* trail_fs = "shader/trail.fs";
	Shader trailShader;

	// Force positive = attraction, négative = répulsion
	float K[3][3] = {
	    /* B */          { 0.f,  2.0f, 2.0f },
	    /* V */          { 2.0f, 0.0f,  1.0f },
	    /* R */          {  -1.0f, -1.0f, -1.0f }
	};

	float M[3] = {
		20.e8f, 30.e8f, 15.e8f
	};

	float zoom = 1.f;

	// --- Trails ---
	unsigned int trailVBO = 0, trailVAO = 0;
	std::vector<std::vector<glm::vec3>> trails;   
	int trailMax = 30;                            
	float trailAlpha = 0.6f;                      
	float trailWidth = 1.0f;                      

	std::vector<TrailGPU> trailGPU;               

	float energyTotal = 0;
	float worldSize = 40.0f;
	float mouseStrength = 80.0f;
	int nombreIteration = 0;

	float radiusParticule = 0.09f;

	bool collision = false;


	Particules()
	: shader(shader_vs, shader_fs),
	  trailShader(trail_vs, trail_fs)
	{}

	void createParticules(glm::vec3 foyer1, int n1,
	                      glm::vec3 foyer2, int n2,
	                      glm::vec3 foyer3, int n3)
	{
	    vertices.clear();
	    vertices.reserve(n1 + n2 + n3);

	    static std::mt19937 rng{ std::random_device{}() };
	    std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * 3.1415926535f);
	    std::uniform_real_distribution<float> radiusDist(5.f, 15.f);
	    std::uniform_real_distribution<float> jitter(-0.05f, 0.05f);
	    std::uniform_real_distribution<float> vx0(-2.f, 2.f);
	    std::uniform_real_distribution<float> vy0(-2.f, 2.f);
	    std::uniform_real_distribution<float> vz0(-2.f, 2.f);

	    auto spawn = [&](glm::vec3 foyer, int n, int type, float r, float g, float b)
	    {
	        for (int i = 0; i < n; ++i)
	        {
	            float a = angleDist(rng);
	            float rr = radiusDist(rng);

	            Vertex v{};
	            v.x = foyer.x + rr * std::cos(a) + jitter(rng);
	            v.y = foyer.y + rr * std::sin(a) + jitter(rng);
	            v.z = foyer.z + jitter(rng) * 0.2f;

	            v.r = r; v.g = g; v.b = b;

	            v.vx = vx0(rng); v.vy = vy0(rng); v.vz = vz0(rng);
	            v.type = type;

	            v.mass = M[type];
				v.radius = radiusParticule;
				v.alive = 1;
				v.energy = 0.0f;

	            vertices.push_back(v);
	        }
	    };

	    spawn(foyer1, n1, 0, 0.2f, 0.5f, 1.0f); // bleu
	    spawn(foyer2, n2, 1, 0.2f, 1.0f, 0.2f); // vert
	    spawn(foyer3, n3, 2, 1.0f, 0.2f, 0.2f); // rouge

	    trails.assign(vertices.size(), {});
		for (size_t i = 0; i < vertices.size(); ++i) {
		    trails[i].push_back(glm::vec3(vertices[i].x, vertices[i].y, vertices[i].z));
		}
	}

	void radiusUpdate()
	{
		for (size_t i = 0; i < vertices.size(); ++i) {
			vertices[i].radius = radiusParticule;
		}
	}

	void mergeBodies(int a, int b)
	{
	    Vertex& A = vertices[a];
	    Vertex& B = vertices[b];
	    if (!A.alive || !B.alive) return;

	    float m1 = A.mass;
	    float m2 = B.mass;
	    float m  = m1 + m2;

	    A.x = (A.x*m1 + B.x*m2) / m;
	    A.y = (A.y*m1 + B.y*m2) / m;
	    A.z = (A.z*m1 + B.z*m2) / m;

	    A.vx = (A.vx*m1 + B.vx*m2) / m;
	    A.vy = (A.vy*m1 + B.vy*m2) / m;
	    A.vz = (A.vz*m1 + B.vz*m2) / m;

	    A.mass = m;

	    float baseMass = M[A.type];
	    float baseR    = 0.1f;
	    float growth   = 0.33f;

	    float newR = baseR * powf(A.mass / (baseMass + 1e-9f), growth);

	    A.radius = std::max(A.radius, newR);

	    A.r = (A.r*m1 + B.r*m2) / m;
	    A.g = (A.g*m1 + B.g*m2) / m;
	    A.b = (A.b*m1 + B.b*m2) / m;

	    auto& Ta = trails[a];
	    auto& Tb = trails[b];
	    Ta.insert(Ta.end(), Tb.begin(), Tb.end());
	    if ((int)Ta.size() > trailMax) Ta.erase(Ta.begin(), Ta.end() - trailMax);

	    B.alive = 0;
	}

	void compactDead()
	{
	    std::vector<Vertex> newV;
	    std::vector<std::vector<glm::vec3>> newT;
	    newV.reserve(vertices.size());
	    newT.reserve(trails.size());

	    for (size_t i = 0; i < vertices.size(); ++i)
	    {
	        if (vertices[i].alive)
	        {
	            newV.push_back(vertices[i]);
	            newT.push_back(std::move(trails[i]));
	        }
	    }

	    vertices = std::move(newV);
	    trails   = std::move(newT);
	}


	void simulate(float dt, bool mouseDown, const glm::vec3& mouseWorld)
	{
	    nombreIteration = 0;
	    if (dt <= 0.0f) return;

	    if (dt > 0.02f) dt = 0.02f;

	    const float softening = 0.05f;
	    const float G = 6.67e-11f;

	    int N = (int)vertices.size();
	    if (N == 0) return;

	    std::vector<glm::vec3> acc(N, glm::vec3(0.0f));

	    energyTotal = 0.0f;

	    for (int i = 0; i < N; ++i)
	    {
	        energyTotal += vertices[i].energy;

	        glm::vec3 pi(vertices[i].x, vertices[i].y, vertices[i].z);
	        float mi = vertices[i].mass;

	        for (int j = i + 1; j < N; ++j)
	        {
	            ++nombreIteration;

	            glm::vec3 pj(vertices[j].x, vertices[j].y, vertices[j].z);
	            float mj = vertices[j].mass;

	            glm::vec3 d = pj - pi;
	            float dist2 = glm::dot(d, d) + softening * softening;

	            if (dist2 < 1e-12f) continue;

	            float invDist = 1.0f / std::sqrt(dist2);
	            glm::vec3 dir = d * invDist;

	            float forceMag = G * (mi * mj) / dist2;
	            glm::vec3 F = dir * forceMag;

	            acc[i] += F / mi;
	            acc[j] -= F / mj;
	        }
	    }

	    if (mouseDown)
	    {
	        const float mouseSoft = 0.10f;

	        for (int i = 0; i < N; ++i)
	        {
	            ++nombreIteration;

	            glm::vec3 p(vertices[i].x, vertices[i].y, vertices[i].z);
	            glm::vec3 d = mouseWorld - p;

	            float dist2 = (glm::dot(d, d) + mouseSoft * mouseSoft) * 1e-5;
	            if (dist2 < 1e-12f) continue;

	            float invDist = 1.0f / std::sqrt(dist2);
	            glm::vec3 dir = d * invDist;

	            float mi = vertices[i].mass;
	            glm::vec3 Fm = dir * (mouseStrength / dist2);

	            acc[i] += Fm / mi;
	        }
	    }

	    for (int i = 0; i < N; ++i)
	    {
	        ++nombreIteration;

	        vertices[i].vx += acc[i].x * dt;
	        vertices[i].vy += acc[i].y * dt;
	        vertices[i].vz += acc[i].z * dt;

	        vertices[i].x += vertices[i].vx * dt;
	        vertices[i].y += vertices[i].vy * dt;
	        vertices[i].z += vertices[i].vz * dt;

	        auto bounce = [&](float& p, float& v)
	        {
	            if (p < -worldSize) { p = -worldSize; v = -v; }
	            if (p >  worldSize) { p =  worldSize; v = -v; }
	        };
	        bounce(vertices[i].x, vertices[i].vx);
	        bounce(vertices[i].y, vertices[i].vy);
	        bounce(vertices[i].z, vertices[i].vz);

	        float v2 = vertices[i].vx*vertices[i].vx + vertices[i].vy*vertices[i].vy + vertices[i].vz*vertices[i].vz;
	        vertices[i].energy = 0.5f * vertices[i].mass * v2;

	        auto& tr = trails[i];
	        tr.push_back(glm::vec3(vertices[i].x, vertices[i].y, vertices[i].z));
	        if ((int)tr.size() > trailMax) {
	            tr.erase(tr.begin(), tr.begin() + ((int)tr.size() - trailMax));
	        }
	    }

		for (int i = 0; i < N; ++i)
		{
		    if (!vertices[i].alive) continue;

		    glm::vec3 pi(vertices[i].x, vertices[i].y, vertices[i].z);

		    for (int j = i + 1; j < N; ++j)
		    {
		        if (!vertices[j].alive) continue;

		        glm::vec3 pj(vertices[j].x, vertices[j].y, vertices[j].z);
		        glm::vec3 d  = pj - pi;

		        float rSum  = vertices[i].radius + vertices[j].radius;
		        float dist2 = glm::dot(d, d);

		        if (dist2 <= rSum * rSum && collision)
		        {
		            int a = i, b = j;

		            float mi = vertices[i].mass;
		            float mj = vertices[j].mass;

		            if (mj > mi) { a = j; b = i; }   // absorbeur = plus massif

		            mergeBodies(a, b);
		        }
		    }
		}

	    compactDead();

	    upload();
	}

	// --- Heatmaps XY ---
	int hmW = 512, hmH = 512;
	GLuint texMass = 0, texEnergy = 0;

	std::vector<float> massGrid;
	std::vector<float> energyGrid;
	std::vector<unsigned char> rgbMass;
	std::vector<unsigned char> rgbEnergy;

	void initHeatmaps(int W=512, int H=512)
	{
	    hmW = W; hmH = H;
	    massGrid.assign(hmW*hmH, 0.0f);
	    energyGrid.assign(hmW*hmH, 0.0f);
	    rgbMass.assign(hmW*hmH*3, 0);
	    rgbEnergy.assign(hmW*hmH*3, 0);

	    auto makeTex = [&](GLuint& tex){
	        glGenTextures(1, &tex);
	        glBindTexture(GL_TEXTURE_2D, tex);
	        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, hmW, hmH, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
	        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	    };
	    makeTex(texMass);
	    makeTex(texEnergy);
	}

	static inline unsigned char toUC(float x) {
    x = (x < 0.f) ? 0.f : (x > 1.f ? 1.f : x);
    return (unsigned char)std::round(x * 255.f);
	}

	static inline void colormap(float t, unsigned char& R, unsigned char& G, unsigned char& B)
	{
	    t = (t < 0.f) ? 0.f : (t > 1.f ? 1.f : t);

	    float r=0,g=0,b=0;
	    if (t < 0.25f) { r = 0.0f; g = t/0.25f; b = 1.0f; }
	    else if (t < 0.5f){ r = 0.0f; g = 1.0f; b = 1.0f - (t-0.25f)/0.25f; }
	    else if (t < 0.75f){ r = (t-0.5f)/0.25f; g = 1.0f; b = 0.0f; }
	    else { r = 1.0f; g = 1.0f - (t-0.75f)/0.25f; b = 0.0f; }

	    R = toUC(r); G = toUC(g); B = toUC(b);
	}

	void updateHeatmapsXY(float clampMassMax = 0.0f, float clampEnergyMax = 0.0f)
	{
	    if (texMass == 0 || texEnergy == 0) return;

	    std::fill(massGrid.begin(), massGrid.end(), 0.0f);
	    std::fill(energyGrid.begin(), energyGrid.end(), 0.0f);

	    // Accumulation
	    for (const auto& p : vertices)
	    {
	        // projection XY -> cellule
	        float nx = (p.x + worldSize/zoom) / (2.0f * worldSize/zoom); // [0..1]
	        float ny = (p.y + worldSize/zoom) / (2.0f * worldSize/zoom);

	        int ix = (int)(nx * (hmW - 1));
	        int iy = (int)(ny * (hmH - 1));

	        if (ix < 0 || ix >= hmW || iy < 0 || iy >= hmH) continue;

	        int idx = iy * hmW + ix;

	        float m = M[p.type];
	        massGrid[idx] += m;
	        energyGrid[idx] += p.energy;
	    }

	    // Auto-scale si clamp=0
	    float maxM = 0.0f, maxE = 0.0f;
	    if (clampMassMax <= 0.0f) {
	        for (float v : massGrid) if (v > maxM) maxM = v;
	    } else maxM = clampMassMax;

	    if (clampEnergyMax <= 0.0f) {
	        for (float v : energyGrid) if (v > maxE) maxE = v;
	    } else maxE = clampEnergyMax;

	    if (maxM < 1e-12f) maxM = 1.0f;
	    if (maxE < 1e-12f) maxE = 1.0f;

	    // Convert -> RGB + upload textures
	    for (int i = 0; i < hmW*hmH; ++i)
	    {
	        float tm = massGrid[i] / maxM;
	        float te = energyGrid[i] / maxE;

	        unsigned char r,g,b;

	        colormap(tm, r,g,b);
	        rgbMass[3*i+0] = r; rgbMass[3*i+1] = g; rgbMass[3*i+2] = b;

	        colormap(te, r,g,b);
	        rgbEnergy[3*i+0] = r; rgbEnergy[3*i+1] = g; rgbEnergy[3*i+2] = b;
	    }

	    glBindTexture(GL_TEXTURE_2D, texMass);
	    glTexSubImage2D(GL_TEXTURE_2D, 0, 0,0, hmW,hmH, GL_RGB, GL_UNSIGNED_BYTE, rgbMass.data());

	    glBindTexture(GL_TEXTURE_2D, texEnergy);
	    glTexSubImage2D(GL_TEXTURE_2D, 0, 0,0, hmW,hmH, GL_RGB, GL_UNSIGNED_BYTE, rgbEnergy.data());
	}

	void upload()
	{
	    glBindBuffer(GL_ARRAY_BUFFER, VBO);
	    glBufferData(GL_ARRAY_BUFFER,
	                 vertices.size() * sizeof(Vertex),
	                 vertices.data(),
	                 GL_DYNAMIC_DRAW);
	}

	void renderTrails()
	{
	    trailGPU.clear();
	    trailGPU.reserve(vertices.size() * trailMax * 2);

	    for (size_t i = 0; i < vertices.size(); ++i)
	    {
	        const auto& tr = trails[i];
	        if (tr.size() < 2) continue;

	        float cr = vertices[i].r, cg = vertices[i].g, cb = vertices[i].b;

	        for (size_t k = 1; k < tr.size(); ++k)
	        {
	            float t = (float)k / (float)(tr.size() - 1);
	            float a = trailAlpha * t;                      

	            const glm::vec3& p0 = tr[k-1];
	            const glm::vec3& p1 = tr[k];

	            trailGPU.push_back({p0.x, p0.y, p0.z, cr, cg, cb, a});
	            trailGPU.push_back({p1.x, p1.y, p1.z, cr, cg, cb, a});
	        }
	    }

	    glLineWidth(trailWidth);

	    glBindVertexArray(trailVAO);
	    glBindBuffer(GL_ARRAY_BUFFER, trailVBO);

	    glBufferData(GL_ARRAY_BUFFER,
	                 trailGPU.size() * sizeof(TrailGPU),
	                 trailGPU.data(),
	                 GL_DYNAMIC_DRAW);

	    glDrawArrays(GL_LINES, 0, (GLsizei)trailGPU.size());

	    glBindVertexArray(0);
	}


	void renderInit()
	{		

	    glGenVertexArrays(1, &VAO);
	    glGenBuffers(1, &VBO);

	    glBindVertexArray(VAO);
	    glBindBuffer(GL_ARRAY_BUFFER, VBO);

	    glBufferData(GL_ARRAY_BUFFER,
	                 vertices.size() * sizeof(Vertex),
	                 vertices.data(),
	                 GL_DYNAMIC_DRAW);

	    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
	    glEnableVertexAttribArray(0);

	    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(3 * sizeof(float)));
	    glEnableVertexAttribArray(1);

	    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, radius));
		glEnableVertexAttribArray(2);

	    glBindVertexArray(0);

	    glGenVertexArrays(1, &trailVAO);
		glGenBuffers(1, &trailVBO);

		glBindVertexArray(trailVAO);
		glBindBuffer(GL_ARRAY_BUFFER, trailVBO);

		glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(TrailGPU), (void*)0);
		glEnableVertexAttribArray(0);

		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(TrailGPU), (void*)(3*sizeof(float)));
		glEnableVertexAttribArray(1);

		glBindVertexArray(0);

	}
	void renderUpdate(){
	    glBindVertexArray(VAO);
	    glBindBuffer(GL_ARRAY_BUFFER, VBO);

	    glBufferSubData(GL_ARRAY_BUFFER, 0,
	                    vertices.size() * sizeof(Vertex),
	                    vertices.data());

	    glDrawArrays(GL_POINTS, 0, (GLsizei)vertices.size());

	    glBindVertexArray(0);
	}

};