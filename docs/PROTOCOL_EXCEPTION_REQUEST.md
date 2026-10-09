# PROTOCOL_EXCEPTION_REQUEST — frozen protocol file identity

Status: `REQUIRED_CLARIFICATION_NOT_APPROVED`

Date: 2026-10-09 (Asia/Bangkok)

Requester: Trang / MobileNetV3-Small Binary V1

## Observation

The shared contract declares frozen protocol SHA-256:

`6cedb8cfed97c07fc19aab4e44d3148b0b931b5bc7ac034505ec3428138b88d0`

The checked-out `protocols/FULL_TRAINING_PROTOCOL_V1.md` at group reference commit
`addaee99f8ba410ac7efd1e6abce34c042561c75` produced byte-level SHA-256:

`21523f6e0c71ac500e311e243e23ea75cfd5402a3acc18b21035d31a9e351c4f`

This may be a different frozen source artifact or an end-of-line representation
difference. It has not been treated as an approved protocol change.

## Decision requested

Please either:

1. provide the exact approved protocol file whose SHA-256 is `6cedb8cf...`; or
2. explicitly approve the repository file and publish the authoritative replacement hash.

Until resolved, benchmark configs remain draft, data access is blocked, and pilot/full
training must not start.
