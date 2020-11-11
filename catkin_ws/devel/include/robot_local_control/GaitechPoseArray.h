// Generated by gencpp from file robot_local_control/GaitechPoseArray.msg
// DO NOT EDIT!


#ifndef ROBOT_LOCAL_CONTROL_MESSAGE_GAITECHPOSEARRAY_H
#define ROBOT_LOCAL_CONTROL_MESSAGE_GAITECHPOSEARRAY_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>
#include <robot_local_control/GaitechPose.h>

namespace robot_local_control
{
template <class ContainerAllocator>
struct GaitechPoseArray_
{
  typedef GaitechPoseArray_<ContainerAllocator> Type;

  GaitechPoseArray_()
    : header()
    , goals()  {
    }
  GaitechPoseArray_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , goals(_alloc)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef std::vector< ::robot_local_control::GaitechPose_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::robot_local_control::GaitechPose_<ContainerAllocator> >::other >  _goals_type;
  _goals_type goals;




  typedef boost::shared_ptr< ::robot_local_control::GaitechPoseArray_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::robot_local_control::GaitechPoseArray_<ContainerAllocator> const> ConstPtr;

}; // struct GaitechPoseArray_

typedef ::robot_local_control::GaitechPoseArray_<std::allocator<void> > GaitechPoseArray;

typedef boost::shared_ptr< ::robot_local_control::GaitechPoseArray > GaitechPoseArrayPtr;
typedef boost::shared_ptr< ::robot_local_control::GaitechPoseArray const> GaitechPoseArrayConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::robot_local_control::GaitechPoseArray_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::robot_local_control::GaitechPoseArray_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace robot_local_control

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': True}
// {'robot_local_control': ['/home/fetch/catkin_ws/src/robot_local_control/msg'], 'std_msgs': ['/opt/ros/indigo/share/std_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::robot_local_control::GaitechPoseArray_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::robot_local_control::GaitechPoseArray_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::robot_local_control::GaitechPoseArray_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::robot_local_control::GaitechPoseArray_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::robot_local_control::GaitechPoseArray_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::robot_local_control::GaitechPoseArray_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::robot_local_control::GaitechPoseArray_<ContainerAllocator> >
{
  static const char* value()
  {
    return "0457324622f09622155c4325cebe2228";
  }

  static const char* value(const ::robot_local_control::GaitechPoseArray_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x0457324622f09622ULL;
  static const uint64_t static_value2 = 0x155c4325cebe2228ULL;
};

template<class ContainerAllocator>
struct DataType< ::robot_local_control::GaitechPoseArray_<ContainerAllocator> >
{
  static const char* value()
  {
    return "robot_local_control/GaitechPoseArray";
  }

  static const char* value(const ::robot_local_control::GaitechPoseArray_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::robot_local_control::GaitechPoseArray_<ContainerAllocator> >
{
  static const char* value()
  {
    return "Header header\n\
GaitechPose[] goals\n\
================================================================================\n\
MSG: std_msgs/Header\n\
# Standard metadata for higher-level stamped data types.\n\
# This is generally used to communicate timestamped data \n\
# in a particular coordinate frame.\n\
# \n\
# sequence ID: consecutively increasing ID \n\
uint32 seq\n\
#Two-integer timestamp that is expressed as:\n\
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n\
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n\
# time-handling sugar is provided by the client library\n\
time stamp\n\
#Frame this data is associated with\n\
# 0: no frame\n\
# 1: global frame\n\
string frame_id\n\
\n\
================================================================================\n\
MSG: robot_local_control/GaitechPose\n\
float64 locationX\n\
float64 locationY\n\
float64 orientation\n\
";
  }

  static const char* value(const ::robot_local_control::GaitechPoseArray_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::robot_local_control::GaitechPoseArray_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.goals);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct GaitechPoseArray_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::robot_local_control::GaitechPoseArray_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::robot_local_control::GaitechPoseArray_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "goals[]" << std::endl;
    for (size_t i = 0; i < v.goals.size(); ++i)
    {
      s << indent << "  goals[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::robot_local_control::GaitechPose_<ContainerAllocator> >::stream(s, indent + "    ", v.goals[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // ROBOT_LOCAL_CONTROL_MESSAGE_GAITECHPOSEARRAY_H
