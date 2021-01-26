// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: Minimap.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_Minimap_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_Minimap_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3014000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3014000 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_Minimap_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_Minimap_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxiliaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[2]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_Minimap_2eproto;
namespace Transfer {
class Minimap;
class MinimapDefaultTypeInternal;
extern MinimapDefaultTypeInternal _Minimap_default_instance_;
class Minimap_friendPositions;
class Minimap_friendPositionsDefaultTypeInternal;
extern Minimap_friendPositionsDefaultTypeInternal _Minimap_friendPositions_default_instance_;
}  // namespace Transfer
PROTOBUF_NAMESPACE_OPEN
template<> ::Transfer::Minimap* Arena::CreateMaybeMessage<::Transfer::Minimap>(Arena*);
template<> ::Transfer::Minimap_friendPositions* Arena::CreateMaybeMessage<::Transfer::Minimap_friendPositions>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace Transfer {

// ===================================================================

class Minimap_friendPositions PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:Transfer.Minimap.friendPositions) */ {
 public:
  inline Minimap_friendPositions() : Minimap_friendPositions(nullptr) {}
  virtual ~Minimap_friendPositions();

  Minimap_friendPositions(const Minimap_friendPositions& from);
  Minimap_friendPositions(Minimap_friendPositions&& from) noexcept
    : Minimap_friendPositions() {
    *this = ::std::move(from);
  }

  inline Minimap_friendPositions& operator=(const Minimap_friendPositions& from) {
    CopyFrom(from);
    return *this;
  }
  inline Minimap_friendPositions& operator=(Minimap_friendPositions&& from) noexcept {
    if (GetArena() == from.GetArena()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const Minimap_friendPositions& default_instance();

  static inline const Minimap_friendPositions* internal_default_instance() {
    return reinterpret_cast<const Minimap_friendPositions*>(
               &_Minimap_friendPositions_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(Minimap_friendPositions& a, Minimap_friendPositions& b) {
    a.Swap(&b);
  }
  inline void Swap(Minimap_friendPositions* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(Minimap_friendPositions* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline Minimap_friendPositions* New() const final {
    return CreateMaybeMessage<Minimap_friendPositions>(nullptr);
  }

  Minimap_friendPositions* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<Minimap_friendPositions>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const Minimap_friendPositions& from);
  void MergeFrom(const Minimap_friendPositions& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(Minimap_friendPositions* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "Transfer.Minimap.friendPositions";
  }
  protected:
  explicit Minimap_friendPositions(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_Minimap_2eproto);
    return ::descriptor_table_Minimap_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kXFieldNumber = 1,
    kYFieldNumber = 2,
  };
  // int32 x = 1;
  void clear_x();
  ::PROTOBUF_NAMESPACE_ID::int32 x() const;
  void set_x(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_x() const;
  void _internal_set_x(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // int32 y = 2;
  void clear_y();
  ::PROTOBUF_NAMESPACE_ID::int32 y() const;
  void set_y(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_y() const;
  void _internal_set_y(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // @@protoc_insertion_point(class_scope:Transfer.Minimap.friendPositions)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::int32 x_;
  ::PROTOBUF_NAMESPACE_ID::int32 y_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_Minimap_2eproto;
};
// -------------------------------------------------------------------

class Minimap PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:Transfer.Minimap) */ {
 public:
  inline Minimap() : Minimap(nullptr) {}
  virtual ~Minimap();

  Minimap(const Minimap& from);
  Minimap(Minimap&& from) noexcept
    : Minimap() {
    *this = ::std::move(from);
  }

  inline Minimap& operator=(const Minimap& from) {
    CopyFrom(from);
    return *this;
  }
  inline Minimap& operator=(Minimap&& from) noexcept {
    if (GetArena() == from.GetArena()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const Minimap& default_instance();

  static inline const Minimap* internal_default_instance() {
    return reinterpret_cast<const Minimap*>(
               &_Minimap_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(Minimap& a, Minimap& b) {
    a.Swap(&b);
  }
  inline void Swap(Minimap* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(Minimap* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline Minimap* New() const final {
    return CreateMaybeMessage<Minimap>(nullptr);
  }

  Minimap* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<Minimap>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const Minimap& from);
  void MergeFrom(const Minimap& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(Minimap* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "Transfer.Minimap";
  }
  protected:
  explicit Minimap(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_Minimap_2eproto);
    return ::descriptor_table_Minimap_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  typedef Minimap_friendPositions friendPositions;

  // accessors -------------------------------------------------------

  enum : int {
    kFPositionsFieldNumber = 2,
    kEPositionsFieldNumber = 4,
    kFriendNumberFieldNumber = 1,
    kEnemyNumberFieldNumber = 3,
  };
  // repeated .Transfer.Minimap.friendPositions fPositions = 2;
  int fpositions_size() const;
  private:
  int _internal_fpositions_size() const;
  public:
  void clear_fpositions();
  ::Transfer::Minimap_friendPositions* mutable_fpositions(int index);
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::Transfer::Minimap_friendPositions >*
      mutable_fpositions();
  private:
  const ::Transfer::Minimap_friendPositions& _internal_fpositions(int index) const;
  ::Transfer::Minimap_friendPositions* _internal_add_fpositions();
  public:
  const ::Transfer::Minimap_friendPositions& fpositions(int index) const;
  ::Transfer::Minimap_friendPositions* add_fpositions();
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::Transfer::Minimap_friendPositions >&
      fpositions() const;

  // repeated .Transfer.Minimap.friendPositions ePositions = 4;
  int epositions_size() const;
  private:
  int _internal_epositions_size() const;
  public:
  void clear_epositions();
  ::Transfer::Minimap_friendPositions* mutable_epositions(int index);
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::Transfer::Minimap_friendPositions >*
      mutable_epositions();
  private:
  const ::Transfer::Minimap_friendPositions& _internal_epositions(int index) const;
  ::Transfer::Minimap_friendPositions* _internal_add_epositions();
  public:
  const ::Transfer::Minimap_friendPositions& epositions(int index) const;
  ::Transfer::Minimap_friendPositions* add_epositions();
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::Transfer::Minimap_friendPositions >&
      epositions() const;

  // int32 friendNumber = 1;
  void clear_friendnumber();
  ::PROTOBUF_NAMESPACE_ID::int32 friendnumber() const;
  void set_friendnumber(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_friendnumber() const;
  void _internal_set_friendnumber(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // int32 enemyNumber = 3;
  void clear_enemynumber();
  ::PROTOBUF_NAMESPACE_ID::int32 enemynumber() const;
  void set_enemynumber(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_enemynumber() const;
  void _internal_set_enemynumber(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // @@protoc_insertion_point(class_scope:Transfer.Minimap)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::Transfer::Minimap_friendPositions > fpositions_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::Transfer::Minimap_friendPositions > epositions_;
  ::PROTOBUF_NAMESPACE_ID::int32 friendnumber_;
  ::PROTOBUF_NAMESPACE_ID::int32 enemynumber_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_Minimap_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// Minimap_friendPositions

// int32 x = 1;
inline void Minimap_friendPositions::clear_x() {
  x_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 Minimap_friendPositions::_internal_x() const {
  return x_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 Minimap_friendPositions::x() const {
  // @@protoc_insertion_point(field_get:Transfer.Minimap.friendPositions.x)
  return _internal_x();
}
inline void Minimap_friendPositions::_internal_set_x(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  x_ = value;
}
inline void Minimap_friendPositions::set_x(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_x(value);
  // @@protoc_insertion_point(field_set:Transfer.Minimap.friendPositions.x)
}

// int32 y = 2;
inline void Minimap_friendPositions::clear_y() {
  y_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 Minimap_friendPositions::_internal_y() const {
  return y_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 Minimap_friendPositions::y() const {
  // @@protoc_insertion_point(field_get:Transfer.Minimap.friendPositions.y)
  return _internal_y();
}
inline void Minimap_friendPositions::_internal_set_y(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  y_ = value;
}
inline void Minimap_friendPositions::set_y(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_y(value);
  // @@protoc_insertion_point(field_set:Transfer.Minimap.friendPositions.y)
}

// -------------------------------------------------------------------

// Minimap

// int32 friendNumber = 1;
inline void Minimap::clear_friendnumber() {
  friendnumber_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 Minimap::_internal_friendnumber() const {
  return friendnumber_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 Minimap::friendnumber() const {
  // @@protoc_insertion_point(field_get:Transfer.Minimap.friendNumber)
  return _internal_friendnumber();
}
inline void Minimap::_internal_set_friendnumber(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  friendnumber_ = value;
}
inline void Minimap::set_friendnumber(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_friendnumber(value);
  // @@protoc_insertion_point(field_set:Transfer.Minimap.friendNumber)
}

// repeated .Transfer.Minimap.friendPositions fPositions = 2;
inline int Minimap::_internal_fpositions_size() const {
  return fpositions_.size();
}
inline int Minimap::fpositions_size() const {
  return _internal_fpositions_size();
}
inline void Minimap::clear_fpositions() {
  fpositions_.Clear();
}
inline ::Transfer::Minimap_friendPositions* Minimap::mutable_fpositions(int index) {
  // @@protoc_insertion_point(field_mutable:Transfer.Minimap.fPositions)
  return fpositions_.Mutable(index);
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::Transfer::Minimap_friendPositions >*
Minimap::mutable_fpositions() {
  // @@protoc_insertion_point(field_mutable_list:Transfer.Minimap.fPositions)
  return &fpositions_;
}
inline const ::Transfer::Minimap_friendPositions& Minimap::_internal_fpositions(int index) const {
  return fpositions_.Get(index);
}
inline const ::Transfer::Minimap_friendPositions& Minimap::fpositions(int index) const {
  // @@protoc_insertion_point(field_get:Transfer.Minimap.fPositions)
  return _internal_fpositions(index);
}
inline ::Transfer::Minimap_friendPositions* Minimap::_internal_add_fpositions() {
  return fpositions_.Add();
}
inline ::Transfer::Minimap_friendPositions* Minimap::add_fpositions() {
  // @@protoc_insertion_point(field_add:Transfer.Minimap.fPositions)
  return _internal_add_fpositions();
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::Transfer::Minimap_friendPositions >&
Minimap::fpositions() const {
  // @@protoc_insertion_point(field_list:Transfer.Minimap.fPositions)
  return fpositions_;
}

// int32 enemyNumber = 3;
inline void Minimap::clear_enemynumber() {
  enemynumber_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 Minimap::_internal_enemynumber() const {
  return enemynumber_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 Minimap::enemynumber() const {
  // @@protoc_insertion_point(field_get:Transfer.Minimap.enemyNumber)
  return _internal_enemynumber();
}
inline void Minimap::_internal_set_enemynumber(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  enemynumber_ = value;
}
inline void Minimap::set_enemynumber(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_enemynumber(value);
  // @@protoc_insertion_point(field_set:Transfer.Minimap.enemyNumber)
}

// repeated .Transfer.Minimap.friendPositions ePositions = 4;
inline int Minimap::_internal_epositions_size() const {
  return epositions_.size();
}
inline int Minimap::epositions_size() const {
  return _internal_epositions_size();
}
inline void Minimap::clear_epositions() {
  epositions_.Clear();
}
inline ::Transfer::Minimap_friendPositions* Minimap::mutable_epositions(int index) {
  // @@protoc_insertion_point(field_mutable:Transfer.Minimap.ePositions)
  return epositions_.Mutable(index);
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::Transfer::Minimap_friendPositions >*
Minimap::mutable_epositions() {
  // @@protoc_insertion_point(field_mutable_list:Transfer.Minimap.ePositions)
  return &epositions_;
}
inline const ::Transfer::Minimap_friendPositions& Minimap::_internal_epositions(int index) const {
  return epositions_.Get(index);
}
inline const ::Transfer::Minimap_friendPositions& Minimap::epositions(int index) const {
  // @@protoc_insertion_point(field_get:Transfer.Minimap.ePositions)
  return _internal_epositions(index);
}
inline ::Transfer::Minimap_friendPositions* Minimap::_internal_add_epositions() {
  return epositions_.Add();
}
inline ::Transfer::Minimap_friendPositions* Minimap::add_epositions() {
  // @@protoc_insertion_point(field_add:Transfer.Minimap.ePositions)
  return _internal_add_epositions();
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::Transfer::Minimap_friendPositions >&
Minimap::epositions() const {
  // @@protoc_insertion_point(field_list:Transfer.Minimap.ePositions)
  return epositions_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace Transfer

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_Minimap_2eproto
