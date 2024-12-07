#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <memory>

using namespace Eigen;
using namespace std;

class VertexAttributes
{
	public:
	VertexAttributes(float x = 0, float y = 0, float z = 0, float w = 1)
	{
		position << x,y,z,w;
	}

    // Interpolates the vertex attributes
    static VertexAttributes interpolate(
        const VertexAttributes& a,
        const VertexAttributes& b,
        const VertexAttributes& c,
        const float alpha, 
        const float beta, 
        const float gamma
    ) 
    {

		VertexAttributes r;
		r.position = alpha * (a.position / a.position[3]) + 
		beta * (b.position / b.position[3]) + gamma * (c.position / c.position[3]);

		// no compute of per face normal and vertex normal

		return r;
    }

	Vector4f position;
	Vector4f normal;
};

class FragmentAttributes
{
	public:
	FragmentAttributes(float r = 0, float g = 0, float b = 0, float a = 1)
	{
		color << r,g,b,a;
	}

	Vector4f color;
	Vector4f position;
};

class FrameBufferAttributes
{
	public:
	FrameBufferAttributes(uint8_t r = 0, uint8_t g = 0, uint8_t b = 0, uint8_t a = 255)
	{
		color << r,g,b,a;
	}

	Matrix<uint8_t,4,1> color;
	float depth;
};

class UniformAttributes
{
	public:

	Vector4f color;

	// Scene info
	Vector3f ambient_light;			  	  

	// Matrices 
    Matrix4f projection_matrix;	      	 
	Matrix4f ortographic_matrix;      	  
	Matrix4f model_matrix;
	Matrix4f camera_matrix;  			  			      
 
	struct Material {
		Vector4f color_mat;
	};

	struct Light {
		Vector3f Liposition;
		Vector4f intensity;
	};

	struct Camera {
		bool is_perspective;
		float field_of_view; 		// between 0 and PI
		float focal_lenght;        
		Vector3f UpVector;
		Vector3f position;
		float far;	
		float near;	
	};

	struct Object {
	Material material;
	virtual ~Object() = default; // Classes with virtual methods should have a virtual destructor!
	};

	// We use smart pointers to hold objects as this is a virtual class
	typedef shared_ptr<Object> ObjectPtr;


	struct Mesh : public Object {

		MatrixXd vertices;                      // n x 3 matrix (n points)
		MatrixXi facets;                        // m x 3 matrix (m triangles)

		Mesh() = default;                       // Default empty constructor
		virtual ~Mesh() = default;
	};

	// List of Material , lights and objects
	Camera camera;
	vector<Material> materials;
	vector<Light> lights;
	vector<ObjectPtr> objects;
};