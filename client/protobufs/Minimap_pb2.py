# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: Minimap.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='Minimap.proto',
  package='Transfer',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\rMinimap.proto\x12\x08Transfer\"\xcb\x01\n\x07Minimap\x12\x14\n\x0c\x66riendNumber\x18\x01 \x01(\x05\x12\x35\n\nfPositions\x18\x02 \x03(\x0b\x32!.Transfer.Minimap.friendPositions\x12\x13\n\x0b\x65nemyNumber\x18\x03 \x01(\x05\x12\x35\n\nePositions\x18\x04 \x03(\x0b\x32!.Transfer.Minimap.friendPositions\x1a\'\n\x0f\x66riendPositions\x12\t\n\x01x\x18\x01 \x01(\x05\x12\t\n\x01y\x18\x02 \x01(\x05\x62\x06proto3'
)




_MINIMAP_FRIENDPOSITIONS = _descriptor.Descriptor(
  name='friendPositions',
  full_name='Transfer.Minimap.friendPositions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='Transfer.Minimap.friendPositions.x', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='y', full_name='Transfer.Minimap.friendPositions.y', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=192,
  serialized_end=231,
)

_MINIMAP = _descriptor.Descriptor(
  name='Minimap',
  full_name='Transfer.Minimap',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='friendNumber', full_name='Transfer.Minimap.friendNumber', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='fPositions', full_name='Transfer.Minimap.fPositions', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='enemyNumber', full_name='Transfer.Minimap.enemyNumber', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ePositions', full_name='Transfer.Minimap.ePositions', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_MINIMAP_FRIENDPOSITIONS, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=28,
  serialized_end=231,
)

_MINIMAP_FRIENDPOSITIONS.containing_type = _MINIMAP
_MINIMAP.fields_by_name['fPositions'].message_type = _MINIMAP_FRIENDPOSITIONS
_MINIMAP.fields_by_name['ePositions'].message_type = _MINIMAP_FRIENDPOSITIONS
DESCRIPTOR.message_types_by_name['Minimap'] = _MINIMAP
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Minimap = _reflection.GeneratedProtocolMessageType('Minimap', (_message.Message,), {

  'friendPositions' : _reflection.GeneratedProtocolMessageType('friendPositions', (_message.Message,), {
    'DESCRIPTOR' : _MINIMAP_FRIENDPOSITIONS,
    '__module__' : 'Minimap_pb2'
    # @@protoc_insertion_point(class_scope:Transfer.Minimap.friendPositions)
    })
  ,
  'DESCRIPTOR' : _MINIMAP,
  '__module__' : 'Minimap_pb2'
  # @@protoc_insertion_point(class_scope:Transfer.Minimap)
  })
_sym_db.RegisterMessage(Minimap)
_sym_db.RegisterMessage(Minimap.friendPositions)


# @@protoc_insertion_point(module_scope)
