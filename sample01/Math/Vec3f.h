#ifndef OSG_VEC3F
#define OSG_VEC3F 1

#include "device_launch_parameters.h"

class Vec3f
{
    public:
		/** Data type of vector components.*/
		typedef float value_type;

		/** Number of vector components. */
		enum { num_components = 3 };

		value_type _v[3];

        /** Constructor that sets all components of the vector to zero */
		__device__ Vec3f() { _v[0] = 0.0f; _v[1] = 0.0f; _v[2] = 0.0f; }
		__device__ Vec3f(value_type x, value_type y, value_type z) { _v[0] = x; _v[1] = y; _v[2] = z; }

		__device__  inline bool operator == (const Vec3f& v) const { return _v[0] == v._v[0] && _v[1] == v._v[1] && _v[2] == v._v[2]; }

		__device__ inline bool operator != (const Vec3f& v) const { return _v[0] != v._v[0] || _v[1] != v._v[1] || _v[2] != v._v[2]; }

		__device__ inline bool operator <  (const Vec3f& v) const
        {
            if (_v[0]<v._v[0]) return true;
            else if (_v[0]>v._v[0]) return false;
            else if (_v[1]<v._v[1]) return true;
            else if (_v[1]>v._v[1]) return false;
            else return (_v[2]<v._v[2]);
        }

		__device__ inline value_type* ptr() { return _v; }
		__device__ inline const value_type* ptr() const { return _v; }

		__device__ inline void set(value_type x, value_type y, value_type z)
        {
            _v[0]=x; _v[1]=y; _v[2]=z;
        }

		__device__ inline void set(const Vec3f& rhs)
        {
            _v[0]=rhs._v[0]; _v[1]=rhs._v[1]; _v[2]=rhs._v[2];
        }

		__device__ inline value_type& operator [] (int i) { return _v[i]; }
		__device__  inline value_type operator [] (int i) const { return _v[i]; }

		__device__ inline value_type& x() { return _v[0]; }
		__device__ inline value_type& y() { return _v[1]; }
		__device__ inline value_type& z() { return _v[2]; }

		__device__ inline value_type x() const { return _v[0]; }
		__device__ inline value_type y() const { return _v[1]; }
		__device__ inline value_type z() const { return _v[2]; }

        /** Returns true if all components have values that are not NaN. */
		//__device__ inline bool valid() const { return !isNaN(); }
        /** Returns true if at least one component has value NaN. */
		//__device__ inline bool isNaN() const { return osg::isNaN(_v[0]) || osg::isNaN(_v[1]) || osg::isNaN(_v[2]); }

        /** Dot product. */
		__device__ inline value_type operator * (const Vec3f& rhs) const
        {
            return _v[0]*rhs._v[0]+_v[1]*rhs._v[1]+_v[2]*rhs._v[2];
        }

        /** Cross product. */
		__device__ inline const Vec3f operator ^ (const Vec3f& rhs) const
        {
            return Vec3f(_v[1]*rhs._v[2]-_v[2]*rhs._v[1],
                         _v[2]*rhs._v[0]-_v[0]*rhs._v[2] ,
                         _v[0]*rhs._v[1]-_v[1]*rhs._v[0]);
        }

        /** Multiply by scalar. */
		__device__ inline const Vec3f operator * (value_type rhs) const
        {
            return Vec3f(_v[0]*rhs, _v[1]*rhs, _v[2]*rhs);
        }

        /** Unary multiply by scalar. */
		__device__ inline Vec3f& operator *= (value_type rhs)
        {
            _v[0]*=rhs;
            _v[1]*=rhs;
            _v[2]*=rhs;
            return *this;
        }

        /** Divide by scalar. */
		__device__ inline const Vec3f operator / (value_type rhs) const
        {
            return Vec3f(_v[0]/rhs, _v[1]/rhs, _v[2]/rhs);
        }

        /** Unary divide by scalar. */
		__device__ inline Vec3f& operator /= (value_type rhs)
        {
            _v[0]/=rhs;
            _v[1]/=rhs;
            _v[2]/=rhs;
            return *this;
        }

        /** Binary vector add. */
		__device__ inline const Vec3f operator + (const Vec3f& rhs) const
        {
            return Vec3f(_v[0]+rhs._v[0], _v[1]+rhs._v[1], _v[2]+rhs._v[2]);
        }

        /** Unary vector add. Slightly more efficient because no temporary
          * intermediate object.
        */
		__device__ inline Vec3f& operator += (const Vec3f& rhs)
        {
            _v[0] += rhs._v[0];
            _v[1] += rhs._v[1];
            _v[2] += rhs._v[2];
            return *this;
        }

        /** Binary vector subtract. */
		__device__ inline const Vec3f operator - (const Vec3f& rhs) const
        {
            return Vec3f(_v[0]-rhs._v[0], _v[1]-rhs._v[1], _v[2]-rhs._v[2]);
        }

        /** Unary vector subtract. */
		__device__ inline Vec3f& operator -= (const Vec3f& rhs)
        {
            _v[0]-=rhs._v[0];
            _v[1]-=rhs._v[1];
            _v[2]-=rhs._v[2];
            return *this;
        }

        /** Negation operator. Returns the negative of the Vec3f. */
		__device__ inline const Vec3f operator - () const
        {
            return Vec3f (-_v[0], -_v[1], -_v[2]);
        }

        /** Length of the vector = sqrt( vec . vec ) */
		__device__ inline value_type length() const
        {
            return sqrtf( _v[0]*_v[0] + _v[1]*_v[1] + _v[2]*_v[2] );
        }

        /** Length squared of the vector = vec . vec */
		__device__ inline value_type length2() const
        {
            return _v[0]*_v[0] + _v[1]*_v[1] + _v[2]*_v[2];
        }

        /** Normalize the vector so that it has length unity.
          * Returns the previous length of the vector.
        */
		__device__ inline value_type normalize()
        {
            value_type norm = Vec3f::length();
            if (norm>0.0)
            {
                value_type inv = 1.0f/norm;
                _v[0] *= inv;
                _v[1] *= inv;
                _v[2] *= inv;
            }
            return( norm );
        }

};    // end of class Vec3f

#endif

