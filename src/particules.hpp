#pragma once 
#include <iostream>
#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "shader.hpp"
#include <random>
#include <cmath>
#include <cstddef>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#ifdef USE_OMP
  #include <omp.h>
#endif

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

	float M[3] = {
		1.e9f, 1.e9f, 1.e9f
	};

	float zoom = 1.f;

	float v0 = 0.f;

	// --- Trails ---
	unsigned int trailVBO = 0, trailVAO = 0;
	std::vector<std::vector<glm::vec3>> trails;   
	int trailMax = 30;                            
	float trailAlpha = 0.6f;                      
	float trailWidth = 1.0f;                      

	std::vector<TrailGPU> trailGPU;               

	float energyTotal = 0;
	float massTotal = 0;
	float worldSize = 200.0f;
	float mouseStrength = 80.0f;

	float radiusParticule = 0.09f;

	bool collision = true;

	bool mouseEffect = false;

	// --- Dislocation (tunable) ---

	bool enableDislocation = true;

	float breakStrength = 0.35f;
	float fragEnergyShare = 0.60f;
	int   maxFragmentsPerCollision = 8;

	float minMassRatio = 0.1f;
	float separationFactor = 1.25f;

	std::vector<glm::vec3> acc;
	std::vector<uint8_t> touched;

	size_t gpuCapacityBytes = 0;
	size_t trailCapacityBytes = 0;

	// --- SATURNE SIMULATION --- 

	struct BodyPreset {
    std::string name;
    float mass;      // masse "simulation" (pas en kg SI si tu veux pas)
    float radius;    // rayon visuel/collision
    glm::vec3 color;
	};

	// ----------------------------
	// REAL DATA (mètres)
	// ----------------------------
	 float RsatReal  = 5.8232e7f;
	 float RinReal   = 7.0e7f;
	 float RoutReal  = 1.4e8f;
	 float ThickReal = 2.0e6f;

	// ----------------------------
	// SCALE automatique
	// ----------------------------
	float saturnRadiusU = worldSize * 0.10f;   // 10% du monde (beau visuellement)
	float scale = saturnRadiusU / RsatReal;

	// ----------------------------
	// CONVERSION unités simulation
	// ----------------------------
	float ringInner     = RinReal   * scale;
	float ringOuter     = RoutReal  * scale;
	float ringThickness = ThickReal * scale;

	// masses → on ne garde PAS les vraies masses (sinon instable)
	float MsatSim = 1.0e12f;
	float ringMassTotal = (float)(scale * (double)MsatSim);

	BodyPreset saturnPreset = {
	    "Saturn",
	    MsatSim,
	    saturnRadiusU,
	    {0.85f, 0.78f, 0.60f}
	};

	Particules()
	: shader(shader_vs, shader_fs),
	  trailShader(trail_vs, trail_fs)
	{}

	float radiusFromMass(float mass, int type) const
	{
	    float baseMass = M[type];
	    float baseR    = radiusParticule;   // ou 0.1f
	    float growth   = 0.33f;             // densité constante
	    float r = baseR * powf(mass / (baseMass + 1e-9f), growth);
	    return r;
	}

	glm::vec3 velOf(const Vertex& v) const { return {v.vx, v.vy, v.vz}; }
	void setVel(Vertex& v, const glm::vec3& w) { v.vx=w.x; v.vy=w.y; v.vz=w.z; }

	void createParticules(glm::vec3 foyer1, int n1,
	                      glm::vec3 foyer2, int n2,
	                      glm::vec3 foyer3, int n3)
	{
	    vertices.clear();
	    vertices.reserve(n1 + n2 + n3);

	    static std::mt19937 rng{ std::random_device{}() };
	    std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * 3.1415926535f);
	    std::uniform_real_distribution<float> radiusDist(0.f, 50.f);
	    std::uniform_real_distribution<float> jitter(-0.05f, 0.05f);
	    std::uniform_real_distribution<float> vx0(-v0, v0);
	    std::uniform_real_distribution<float> vy0(-v0, v0);
	    std::uniform_real_distribution<float> vz0(-v0, v0);

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

		spawn(foyer1, n1, 0, 0.4f, 0.7f, 1.0f);   // bleu nébuleuse
		spawn(foyer2, n2, 1, 0.6f, 1.0f, 0.8f);   // vert menthe cosmique
		spawn(foyer3, n3, 2, 1.0f, 0.6f, 0.4f);   // orange stellaire

	    trails.assign(vertices.size(), {});
		for (size_t i = 0; i < vertices.size(); ++i) {
		    trails[i].push_back(glm::vec3(vertices[i].x, vertices[i].y, vertices[i].z));
		}
	}

	static float rand01(std::mt19937& rng) {
	    static std::uniform_real_distribution<float> U(0.f, 1.f);
	    return U(rng);
	}

	// Tirage "power-law" sur [mMin, mMax] : dN/dm ~ m^{-alpha}
	static float samplePowerLaw(std::mt19937& rng, float mMin, float mMax, float alpha)
	{
	    float u = rand01(rng);
	    if (std::abs(alpha - 1.f) < 1e-6f) {
	        // cas alpha ~ 1 : log-uniform
	        return mMin * std::exp(std::log(mMax / mMin) * u);
	    }
	    // inversion CDF
	    float a = 1.f - alpha;
	    float x1 = std::pow(mMin, a);
	    float x2 = std::pow(mMax, a);
	    float x  = x1 + (x2 - x1) * u;
	    return std::pow(x, 1.f / a);
	}

	void createSaturnSystem(glm::vec3 center, int nRings)
	{
	    vertices.clear();
	    vertices.reserve(1 + nRings);

	    static std::mt19937 rng{ std::random_device{}() };

	    // Distributions
	    std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * 3.1415926535f);
	    std::uniform_real_distribution<float> radDist(ringInner, ringOuter);
	    std::uniform_real_distribution<float> zDist(-ringThickness, ringThickness);
	    std::uniform_real_distribution<float> jitterXY(-0.02f, 0.02f);

	    // --- 1) Saturne (corps central) ---
	    {
	        Vertex s{};
	        s.x = center.x; s.y = center.y; s.z = center.z;

	        s.r = saturnPreset.color.r;
	        s.g = saturnPreset.color.g;
	        s.b = saturnPreset.color.b;

	        s.vx = 0.f; s.vy = 0.f; s.vz = 0.f;
	        s.type = 0;                 // type arbitraire
	        s.mass = saturnPreset.mass; // masse Saturne (ou masse "sim")
	        s.radius = saturnPreset.radius;
	        s.alive = 1;
	        s.energy = 0.0f;

	        vertices.push_back(s);
	    }

	    // Constante gravitation (celle que tu utilises déjà)
	    const float G = 6.67e-11f;

	    // --- masses anneaux non uniformes ---
		std::vector<float> mRaw(nRings);
		mRaw.reserve(nRings);

		// Paramètres : à ajuster
		float mMin = ringMassTotal * 1e-6f / std::max(1, nRings); // très petites
		float mMax = ringMassTotal * 5e-2f / std::max(1, nRings); // quelques "gros"
		float alpha = 1.8f; // ~ astéroïdes: souvent entre 1.6 et 2.2 (visuel OK)

		double sumRaw = 0.0;
		for (int i = 0; i < nRings; ++i) {
		    float m = samplePowerLaw(rng, mMin, mMax, alpha);
		    mRaw[i] = m;
		    sumRaw += (double)m;
		}

		// Renormalisation pour que somme(m) = ringMassTotal
		float scaleM = (sumRaw > 0.0) ? (ringMassTotal / (float)sumRaw) : 0.0f;

	    // --- 2) Anneaux : particules en orbite quasi circulaire ---
	    for (int i = 0; i < nRings; ++i)
	    {
	        float a  = angleDist(rng);
	        float rr = radDist(rng);

	        // position dans un disque (plan XY)
	        float x = center.x + rr * std::cos(a) + jitterXY(rng);
	        float y = center.y + rr * std::sin(a) + jitterXY(rng);
	        float z = center.z + zDist(rng);

	        // Vitesse orbitale circulaire v = sqrt(G*M/r)
	        // (si tu es en unités non-SI, ajuste G et masses)
	        float v = std::sqrt(std::max(0.0f, G * saturnPreset.mass / (rr + 1e-6f)));

	        // direction tangentielle (perpendiculaire au rayon)
	        float tx = -std::sin(a);
	        float ty =  std::cos(a);

	        // petite dispersion (anneaux pas parfaitement circulaires)
	        std::uniform_real_distribution<float> dv(-0.02f*v, 0.02f*v);
	        float vv = v + dv(rng);

	        Vertex p{};
	        p.x = x; p.y = y; p.z = z;

	        // couleur anneaux (tu peux faire varier selon rr pour simuler A/B/C)
	        // exemple simple : plus clair vers l'extérieur
	        float t = (rr - ringInner) / (ringOuter - ringInner + 1e-9f);
	        p.r = 0.8f + 0.15f * t;
	        p.g = 0.8f + 0.15f * t;
	        p.b = 0.85f + 0.10f * t;

	        p.vx = vv * tx;
	        p.vy = vv * ty;
	        p.vz = 0.0f;

	        p.type = 1;
	        p.alive = 1;

	        // masse par particule (répartie sur nRings)
	        p.mass = mRaw[i] * scaleM;

			float mRef = ringMassTotal / (float)std::max(1, nRings);
			float k = radiusParticule / std::cbrt(std::max(mRef, 1e-12f));
			p.radius = k * std::cbrt(std::max(p.mass, 1e-12f));

			// clamp pour éviter des monstres / invisibles
			p.radius = std::max(0.25f * radiusParticule, std::min(3.0f * radiusParticule, p.radius));

	        p.energy = 0.5f * p.mass * (p.vx*p.vx + p.vy*p.vy + p.vz*p.vz);

	        vertices.push_back(p);
	    }

	    // Trails
	    trails.assign(vertices.size(), {});
	    for (size_t i = 0; i < vertices.size(); ++i)
	        trails[i].push_back(glm::vec3(vertices[i].x, vertices[i].y, vertices[i].z));
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

	static inline float clampf(float x, float a, float b) {
	    return (x < a) ? a : (x > b ? b : x);
	}

	bool dislocateTarget(int attacker, int target,
	                     const glm::vec3& normal,
	                     float Eimpact,
	                     int& spawnedThisFrame)
	{
	    if (attacker < 0 || target < 0) return false;
	    if (attacker >= (int)vertices.size() || target >= (int)vertices.size()) return false;
	    if (!vertices[attacker].alive || !vertices[target].alive) return false;

	    const Vertex A0 = vertices[attacker];
	    const Vertex B0 = vertices[target];

	    const float G = 6.67e-11f;
	    float Eb = breakStrength * G * (B0.mass * B0.mass) / (B0.radius + 1e-6f);

	    if (Eimpact <= Eb) return false;

	    float surplus = (Eimpact - Eb) / (Eimpact + Eb + 1e-9f); // [0..1)
	    float ejectFrac = clampf(surplus, 1e-6f, 0.99f);         // évite 0 et 100%

	    float ejectMass  = B0.mass * ejectFrac;
	    float remainMass = B0.mass - ejectMass;

	    float minM = minMassRatio * M[B0.type];
	    if (remainMass < minM) {
	        ejectMass = B0.mass;
	        remainMass = 0.0f;
	    }

	    int n = 2 + (int)std::floor(ejectFrac * (maxFragmentsPerCollision - 2));
	    n = std::max(2, std::min(maxFragmentsPerCollision, n));

	    float mf = ejectMass / (float)n;
	    if (mf < minM) {
	        n = (int)std::floor(ejectMass / minM);
	        n = std::max(2, std::min(maxFragmentsPerCollision, n));
	        mf = ejectMass / (float)n;
	        if (mf < minM) return false;
	    }

	    const int maxSpawnPerCollision = maxFragmentsPerCollision;
	    if (spawnedThisFrame + n > maxSpawnPerCollision) return false;

	    float Efrag = fragEnergyShare * (Eimpact - Eb);
	    if (Efrag < 0.0f) Efrag = 0.0f;

	    float vEject = std::sqrt(2.0f * Efrag / (ejectMass + 1e-9f));

	    glm::vec3 vA0 = velOf(A0);
	    glm::vec3 vB0 = velOf(B0);
	    glm::vec3 Pbefore = A0.mass * vA0 + B0.mass * vB0;

	    glm::vec3 vA_after = vA0;

	    // On prépare momentum fragments
	    glm::vec3 Pfrags(0.0f);

	    static std::mt19937 rng{ std::random_device{}() };
	    std::uniform_real_distribution<float> jitter(-0.35f, 0.35f);

	    float Br_after = (remainMass > 0.0f) ? radiusFromMass(remainMass, B0.type) : B0.radius;

	    for (int k = 0; k < n; ++k)
	    {
	        Vertex f{};
	        f.alive = 1;
	        f.type  = B0.type;
	        f.r = B0.r; f.g = B0.g; f.b = B0.b;

	        f.mass   = mf;
	        f.radius = radiusFromMass(f.mass, f.type);

	        float safeDist = ((Br_after > 0 ? Br_after : radiusParticule) + f.radius) * separationFactor;
	        glm::vec3 randOff(jitter(rng), jitter(rng), jitter(rng));
	        glm::vec3 pos = glm::vec3(B0.x, B0.y, B0.z) + normal * safeDist + randOff;

	        f.x = pos.x; f.y = pos.y; f.z = pos.z;

	        glm::vec3 v = vB0 + 1.f/3.f * normal * vEject + randOff * 0.5f;
	        setVel(f, v);

	        float vv2 = glm::dot(v, v);
	        f.energy = 0.5f * f.mass * vv2;

	        vertices.push_back(f);
	        trails.push_back({});
	        trails.back().push_back(glm::vec3(f.x, f.y, f.z));

	        Pfrags += f.mass * v;
	        spawnedThisFrame++;
	    }

	    if (target < (int)vertices.size() && vertices[target].alive)
	    {
	        if (remainMass > 0.0f) {
	            glm::vec3 vB_after = (Pbefore - A0.mass * vA_after - Pfrags) / (remainMass + 1e-9f);

	            vertices[target].mass = remainMass;
	            vertices[target].radius = Br_after;
	            setVel(vertices[target], vB_after);
	        } else {
	            vertices[target].alive = 0;
	        }
	    }

	    return true;
	}


	void simulate(float dt, bool mouseDown, const glm::vec3& mouseWorld)
	{

		int N = (int)vertices.size();
		if ((int)acc.size() < N) acc.resize(N);
		std::fill(acc.begin(), acc.begin() + N, glm::vec3(0.0f));

		if ((int)touched.size() < N) touched.resize(N);
		std::fill(touched.begin(), touched.begin() + N, 0);

		const int maxSpawnPerFrame = 512;
		vertices.reserve(vertices.size() + (size_t)maxSpawnPerFrame);
		trails.reserve(trails.size() + (size_t)maxSpawnPerFrame);

		int spawnedThisFrame = 0;


	    if (dt <= 0.0f) return;

	    if (dt > 0.02f) dt = 0.02f;

	    const float softening = 0.05f;
	    const float G = 6.67e-11f;

	    if (N == 0) return;

		#ifdef USE_OMP
		#pragma omp parallel for schedule(static)
		#endif
		for (int i = 0; i < N; ++i)
		{
		    if (!vertices[i].alive) { acc[i] = glm::vec3(0.0f); continue; }

		    float xi = vertices[i].x, yi = vertices[i].y, zi = vertices[i].z;

		    float axi = 0.f, ayi = 0.f, azi = 0.f;

		    for (int j = 0; j < N; ++j)
		    {
		        if (j == i) continue;
		        if (!vertices[j].alive) continue;

		        float dx = vertices[j].x - xi;
		        float dy = vertices[j].y - yi;
		        float dz = vertices[j].z - zi;

		        float dist2 = dx*dx + dy*dy + dz*dz + softening*softening;
		        if (dist2 < 1e-12f) continue;

		        float invDist = 1.0f / std::sqrt(dist2);
		        float invDist3 = invDist / dist2;

		        float s = G * vertices[j].mass * invDist3; // a = G*mj * r / r^3

		        axi += dx * s;
		        ayi += dy * s;
		        azi += dz * s;
		    }

		    acc[i] = glm::vec3(axi, ayi, azi);
		}

		if (mouseDown && mouseEffect)
		{
		    const float mouseSoft = 0.10f;

		    #ifdef USE_OMP
		    #pragma omp parallel for schedule(static)
		    #endif
		    for (int i = 0; i < N; ++i)
		    {
		        if (!vertices[i].alive) continue;

		        float dx = mouseWorld.x - vertices[i].x;
		        float dy = mouseWorld.y - vertices[i].y;
		        float dz = mouseWorld.z - vertices[i].z;

		        float dist2 = (dx*dx + dy*dy + dz*dz + mouseSoft*mouseSoft) * 1e-5f;
		        if (dist2 < 1e-12f) continue;

		        float invDist = 1.0f / std::sqrt(dist2);
		        float fx = dx * invDist * (mouseStrength / dist2);
		        float fy = dy * invDist * (mouseStrength / dist2);
		        float fz = dz * invDist * (mouseStrength / dist2);

		        float invMi = 1.0f / (vertices[i].mass + 1e-9f);
		        acc[i] += glm::vec3(fx*invMi, fy*invMi, fz*invMi);
		    }
		}

		#ifdef USE_OMP
		#pragma omp parallel for schedule(static)
		#endif
		for (int i = 0; i < N; ++i)
		{
		    if (!vertices[i].alive) continue;

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
				    if (vertices[b].mass > vertices[a].mass) std::swap(a, b); // a = plus massif

				    Vertex& A = vertices[a];
				    Vertex& B = vertices[b];

				    // normale de collision (pour éjection)
				    glm::vec3 nrm = glm::normalize(d + glm::vec3(1e-8f)); // d = pj-pi (si i/j)
				    // si on a swap, assure-toi que nrm pointe de B vers l'extérieur :
				    glm::vec3 pA(A.x,A.y,A.z), pB(B.x,B.y,B.z);
				    nrm = glm::normalize(pB - pA);
				    if (glm::length(nrm) < 1e-12f)
    					nrm = glm::vec3(1,0,0);

					glm::vec3 vA = velOf(A);
					glm::vec3 vB = velOf(B);
					glm::vec3 vRel = vA - vB;
					float mu = (A.mass * B.mass) / (A.mass + B.mass + 1e-9f);

					float vrel_n = glm::dot(vRel, nrm);
					if (vrel_n <= 0.0f) {
					    // pas un impact compressif
					    continue;
					}

					bool didDislocate = false;

					float Eimpact = 0.5f * mu * vrel_n * vrel_n;

					// dislocation seulement si B est nettement plus petit
					float ratio = B.mass / (A.mass + 1e-9f);
					if (enableDislocation && ratio < 0.25f) {
					    didDislocate = dislocateTarget(a, b, nrm, Eimpact, spawnedThisFrame);
					}

				    if (!didDislocate)
				    {
				        mergeBodies(a, b);
				    }

				    break;
				}
		    }
		}

		compactDead();

		energyTotal = 0.0f;
		massTotal   = 0.0f;
		for (size_t i = 0; i < vertices.size(); ++i) {
		    energyTotal += vertices[i].energy;
		    massTotal   += vertices[i].mass;
		}


	    upload();
	}

	// --- Heatmaps XY ---
	int hmW = 256, hmH = 256;
	GLuint texMass = 0, texEnergy = 0;

	std::vector<float> massGrid;
	std::vector<float> energyGrid;
	std::vector<unsigned char> rgbMass;
	std::vector<unsigned char> rgbEnergy;

	void initHeatmaps(int W=256, int H=256)
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

			massGrid[idx] += p.mass;
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

		const int N = hmW * hmH;
		for (int i = 0; i < N; ++i)
		{
		    float tm    = std::log1p(massGrid[i]);
		    float tmMax = std::log1p(maxM);
		    tm = tm / (tmMax > 1e-12f ? tmMax : 1.0f);

		    float te    = std::log1p(energyGrid[i]);
		    float teMax = std::log1p(maxE);
		    te = te / (teMax > 1e-12f ? teMax : 1.0f);

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
	    size_t bytes = vertices.size() * sizeof(Vertex);

	    if (bytes > gpuCapacityBytes) {
	        gpuCapacityBytes = std::max(bytes, gpuCapacityBytes * 2 + 1024);
	        glBufferData(GL_ARRAY_BUFFER, gpuCapacityBytes, nullptr, GL_DYNAMIC_DRAW); // réserve
	    }
	    glBufferSubData(GL_ARRAY_BUFFER, 0, bytes, vertices.data());
	}

	void renderTrails()
	{
	    if (trailVAO == 0 || trailVBO == 0) return;

	    if (trails.size() != vertices.size()) {
		        trails.resize(vertices.size());
		}

	    size_t n = std::min(vertices.size(), trails.size());
	    if (n == 0) return;

	    trailGPU.clear();
	    trailGPU.reserve(n * (size_t)trailMax * 2);

	    for (size_t i = 0; i < n; ++i)
	    {
	        if (!vertices[i].alive) continue;

	        const auto& tr = trails[i];
	        if (tr.size() < 2) continue;

	        float cr = vertices[i].r, cg = vertices[i].g, cb = vertices[i].b;

	        for (size_t k = 1; k < tr.size(); ++k)
	        {
	            float t = (float)k / (float)(tr.size() - 1);
	            float a = trailAlpha * t;

	            const glm::vec3& p0 = tr[k-1];
	            const glm::vec3& p1 = tr[k];

	            trailGPU.push_back({p0.x,p0.y,p0.z, cr,cg,cb, a});
	            trailGPU.push_back({p1.x,p1.y,p1.z, cr,cg,cb, a});
	        }
	    }

	    glLineWidth(trailWidth);

	    glBindVertexArray(trailVAO);
	    glBindBuffer(GL_ARRAY_BUFFER, trailVBO);

	    size_t bytes = trailGPU.size() * sizeof(TrailGPU);
	    if (bytes == 0) { glBindVertexArray(0); return; }

	    if (bytes > trailCapacityBytes) {
	        trailCapacityBytes = std::max(bytes, trailCapacityBytes * 2 + 1024);
	        glBufferData(GL_ARRAY_BUFFER, trailCapacityBytes, nullptr, GL_DYNAMIC_DRAW);
	    }
	    glBufferSubData(GL_ARRAY_BUFFER, 0, bytes, trailGPU.data());

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