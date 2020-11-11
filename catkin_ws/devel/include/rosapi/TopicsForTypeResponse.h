// Generated by gencpp from file rosapi/TopicsForTypeResponse.msg
// DO NOT EDIT!


#ifndef ROSAPI_MESSAGE_TOPICSFORTYPERESPONSE_H
#define ROSAPI_MESSAGE_TOPICSFORTYPERESPONSE_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace rosapi
{
template <class ContainerAllocator>
struct TopicsForTypeResponse_
{
  typedef TopicsForTypeResponse_<ContainerAllocator> Type;

  TopicsForTypeResponse_()
    : topics()  {
    }
  TopicsForTypeResponse_(const ContainerAllocator& _alloc)
    : topics(_alloc)  {
  (void)_alloc;
    }



   typedef std::vector<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > , typename ContainerAllocator::template rebind<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::other >  _topics_type;
  _topics_type topics;




  typedef boost::shared_ptr< ::rosapi::TopicsForTypeResponse_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::rosapi::TopicsForTypeResponse_<ContainerAllocator> const> ConstPtr;

}; // struct TopicsForTypeResponse_

typedef ::rosapi::TopicsForTypeResponse_<std::allocator<void> > TopicsForTypeResponse;

typedef boost::shared_ptr< ::rosapi::TopicsForTypeResponse > TopicsForTypeResponsePtr;
typedef boost::shared_ptr< ::rosapi::TopicsForTypeResponse const> TopicsForTypeResponseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::rosapi::TopicsForTypeResponse_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::rosapi::TopicsForTypeResponse_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace rosapi

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'rosapi': ['/home/fetch/catkin_ws/src/rosbridge_suite/rosapi/msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::rosapi::TopicsForTypeResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::rosapi::TopicsForTypeResponse_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::rosapi::TopicsForTypeResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::rosapi::TopicsForTypeResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rosapi::TopicsForTypeResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rosapi::TopicsForTypeResponse_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::rosapi::TopicsForTypeResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "b0eef9a05d4e829092fc2f2c3c2aad3d";
  }

  static const char* value(const ::rosapi::TopicsForTypeResponse_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xb0eef9a05d4e8290ULL;
  static const uint64_t static_value2 = 0x92fc2f2c3c2aad3dULL;
};

template<class ContainerAllocator>
struct DataType< ::rosapi::TopicsForTypeResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "rosapi/TopicsForTypeResponse";
  }

  static const char* value(const ::rosapi::TopicsForTypeResponse_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::rosapi::TopicsForTypeResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "string[] topics\n\
";
  }

  static const char* value(const ::rosapi::TopicsForTypeResponse_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::rosapi::TopicsForTypeResponse_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.topics);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct TopicsForTypeResponse_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::rosapi::TopicsForTypeResponse_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::rosapi::TopicsForTypeResponse_<ContainerAllocator>& v)
  {
    s << indent << "topics[]" << std::endl;
    for (size_t i = 0; i < v.topics.size(); ++i)
    {
      s << indent << "  topics[" << i << "]: ";
      Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.topics[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // ROSAPI_MESSAGE_TOPICSFORTYPERESPONSE_H
