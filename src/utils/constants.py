"""Constants used throughout the Greek Forge application."""

from typing import Literal

CONTRACT_TYPE_CALL = "CALL"
CONTRACT_TYPE_PUT = "PUT"

ContractType = Literal["CALL", "PUT"]

VALID_CONTRACT_TYPES = (CONTRACT_TYPE_CALL, CONTRACT_TYPE_PUT)
