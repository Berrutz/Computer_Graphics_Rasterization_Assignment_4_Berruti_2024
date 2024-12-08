#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <memory>

#include <iostream>

using namespace Eigen;
using namespace std;

class VertexAttributes
{
	public:


	Vector4f position;
	Vector4f face_normal;
	Vector4f vertex_normal;
	//Vector4f color_mat;

	VertexAttributes(float x = 0, float y = 0, float z = 0, float w = 1,Vector4f normal_face_4d = Vector4f(0,0,0,0),Vector4f normal_vertex_4d = Vector4f(0,0,0,0)) 
	{
		position << x,y,z,w;
		face_normal << normal_face_4d;
		vertex_normal << normal_vertex_4d;
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
		
		

		// Interpolate Normal  
		// Face Normal is the same for each Vertex
		r.face_normal = a.face_normal;  

		// Vertex Normal depend on each 
		r.vertex_normal = alpha * a.vertex_normal + beta * b.vertex_normal + gamma * c.vertex_normal;
 

		return r;
	}
	
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
	float depth = std::numeric_limits<float>::infinity();
};

class UniformAttributes
{
	public:

	// new 
	int frame;
	float rotation_speed_along_axis;
	float translation_speed_toward_camera;
	Vector3f center_of_the_model;
	

	int object_material_index;

	// Scene info	  
	Vector4f ambient_light;			  	  
	Vector3f top_right_corner;           
	float far;		

	// Matrices 
    Matrix4f projection_matrix;	      					
	Matrix4f projection_matrix_inv_transpose;	
	Matrix4f projection_matrix_inverse;                          

	Matrix4f model_matrix;
	Matrix4f model_matrix_inv_transpose;
	Matrix4f camera_matrix;  	
	Matrix4f camera_matrix_inv_transpose;		  			      


	struct Material {

		Vector4f color_mat;
		
		Vector4f ambient_color;
		Vector4f diffuse_color;
		Vector4f specular_color;
		float specular_exponent; 						// Also called "shininess"
	};

	struct Light {
		Vector4f Liposition;
		Vector4f intensity;
	};

	struct Camera {
		bool is_perspective;
		float field_of_view; 	// between 0 and PI
		Vector3f UpVector;
		Vector3f position;
		Vector3f Target;
		float focal_lenght;
		float far , near ;
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
		
		//Mesh(const string &filename,int material_value ,Program &program,UniformAttributes &uniform, FrameBuffer &frameBuffer);
		virtual ~Mesh() = default;
		//virtual bool intersect(const Ray &ray, Intersection &hit) override;
	};

	// List of Material , lights and objects
	Camera camera;
	vector<Material> materials;
	vector<Light> lights;
	vector<ObjectPtr> objects;
};