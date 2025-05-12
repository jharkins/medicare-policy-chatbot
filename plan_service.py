import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlanService:
    """
    Service for loading and querying plan metadata from a JSON file.
    """

    def __init__(self, plans_json_path: Path):
        raw = json.loads(plans_json_path.read_text())
        self._plans: Dict[str, Dict] = {}
        self._hash_to_plan: Dict[str, str] = {}

        for entry in raw:
            plan_id = entry["plan_id"]
            logger.info(f"Loading plan: {plan_id}")
            self._plans[plan_id] = entry
            docs = entry.get("documents", [])
            logger.info(f"Plan {plan_id} has {len(docs)} documents.")
            # map each document hash to this plan
            for doc in docs:
                bh = str(doc["binary_hash"])
                logger.info(f"Mapping binary_hash {bh} to plan {plan_id}")
                self._hash_to_plan[bh] = plan_id
        logger.info(f"Loaded {len(self._plans)} plans.")
        logger.info(f"Mapped {len(self._hash_to_plan)} binary hashes to plans.")

    def list_plans(self) -> List[Dict]:
        """Return full list of plan entries."""
        logger.info("list_plans called")
        return list(self._plans.values())

    def get_plan(self, plan_id: str) -> Optional[Dict]:
        """Get a single plan entry by ID."""
        logger.info(f"get_plan called with plan_id={plan_id}")
        return self._plans.get(plan_id)

    def get_hashes(self, plan_id: str) -> List[str]:
        """Get all binary_hashes associated with a plan."""
        logger.info(f"get_hashes called with plan_id={plan_id}")
        plan = self.get_plan(plan_id)
        if not plan:
            return []
        return [str(doc["binary_hash"]) for doc in plan.get("documents", [])]

    def plan_for_hash(self, binary_hash: str) -> Optional[str]:
        """Get the plan_id for a given document hash."""
        logger.info(f"plan_for_hash called with binary_hash={binary_hash}")
        return self._hash_to_plan.get(binary_hash)

    def get_filename(self, binary_hash: str) -> Optional[str]:
        """Get the filename for a given document hash."""
        logger.info(f"get_filename called with binary_hash={binary_hash}")
        plan_id = self.plan_for_hash(binary_hash)
        if not plan_id:
            return None
        plan = self.get_plan(plan_id)
        if not plan:
            return None
        for doc in plan.get("documents", []):
            if str(doc["binary_hash"]) == str(binary_hash):
                return doc.get("filename")
        return None
