from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from itertools import cycle
import io, math

from hybrid_search import HybridSearcher
from docling_core.types.doc.document import DoclingDocument
from PIL import Image, ImageDraw
from plan_service import PlanService

app = FastAPI()

# Initialize core services
PLAN_JSON_PATH = Path("./plans.json")
DOC_STORE_DIR = Path("./extracted_docs")

searcher = HybridSearcher()
plan_service = PlanService(PLAN_JSON_PATH)


# Request body models
class Box(BaseModel):
    l: float
    t: float
    r: float
    b: float


class AnnotateRequest(BaseModel):
    binary_hash: str
    page: int
    boxes: List[Box]


# Endpoints
@app.get("/api/plans")
def list_plans():
    """Return all loaded plans and their metadata."""
    return {"plans": plan_service.list_plans()}


@app.get("/api/search")
def search(q: str):
    """Perform a text-based search across all documents."""
    return {"result": searcher.search(text=q)}


@app.get("/api/visual_grounding")
def visual_grounding(
    q: str,
    plan_id: Optional[str] = None,
    k: int = Query(3, ge=1, le=10),
):
    """
    Retrieve visual grounding hits for a query, optionally filtered to a specific plan.
    """
    points = searcher.visual_grounding(q, limit=k)

    # Filter by plan if requested
    if plan_id:
        plan = plan_service.get_plan(plan_id)
        if not plan:
            raise HTTPException(404, f"Unknown plan_id '{plan_id}'")
        allowed_hashes = plan_service.get_hashes(plan_id)

        # Safely filter points
        filtered_points = []
        for pt in points:
            if pt.payload is not None:
                origin = pt.payload.get("origin")
                if isinstance(origin, dict):
                    binary_hash = origin.get("binary_hash")
                    if binary_hash is not None and str(binary_hash) in allowed_hashes:
                        filtered_points.append(pt)
        points = filtered_points

    results = []
    for pt in points:
        if pt.payload is None:
            continue

        dl = pt.payload
        origin_data = dl.get("origin")
        if not isinstance(origin_data, dict):
            continue
        binary_hash_value = origin_data.get("binary_hash")
        if binary_hash_value is None:
            continue
        bh = str(binary_hash_value)

        pid = plan_service.plan_for_hash(bh)
        # Ensure plan_meta is a dictionary and pid is not None before calling get_plan
        plan_meta = {}
        if pid is not None:
            plan_meta_candidate = plan_service.get_plan(pid)
            if isinstance(plan_meta_candidate, dict):
                plan_meta = plan_meta_candidate

        provs = []
        doc_items_list = dl.get("doc_items", [])
        if isinstance(doc_items_list, list):
            for item in doc_items_list:
                if isinstance(item, dict):
                    item_provs_list = item.get("prov", [])
                    if isinstance(item_provs_list, list):
                        for p_item in item_provs_list:
                            if (
                                isinstance(p_item, dict)
                                and "page_no" in p_item
                                and "bbox" in p_item
                            ):
                                provs.append(
                                    {"page": p_item["page_no"], "bbox": p_item["bbox"]}
                                )

        page_num = provs[0]["page"] if provs else 0
        boxes = [p["bbox"] for p in provs]

        results.append(
            {
                "text": dl.get("text", dl.get("document", "")),  # Support both new 'text' field and legacy 'document' field
                "binary_hash": bh,
                "plan_id": pid,
                "plan_name": plan_meta.get("plan_name", ""),
                "headings": dl.get("headings", []),
                "provs": provs,
                "annotate_request_body": {
                    "binary_hash": bh,
                    "page": page_num,
                    "boxes": boxes,
                },
                "annotate_endpoint": "/api/annotate_result",
            }
        )

    return {"result": results}


@app.post("/api/annotate_result")
def annotate_result(req: AnnotateRequest):
    """
    Draw bounding boxes on a document page and return a PNG image.
    """
    # Determine JSON path from plan_service
    filename = plan_service.get_filename(req.binary_hash)
    if not filename:
        raise HTTPException(404, f"Document not found for hash '{req.binary_hash}'")
    json_path = DOC_STORE_DIR / filename

    # Load the DoclingDocument
    doc = DoclingDocument.load_from_json(json_path)

    # Load the requested page
    try:
        if doc.pages is None or req.page < 0:
            raise HTTPException(400, f"Page {req.page} invalid or pages not loaded")
        page = doc.pages[req.page]
    except (IndexError, TypeError):
        raise HTTPException(
            400, f"Page {req.page} out of range or invalid pages structure"
        )

    if page.image is None or page.image.pil_image is None:
        raise HTTPException(500, "Page image data is missing")
    img = page.image.pil_image.copy()

    if (
        page.size is None
        or page.size.width is None
        or page.size.height is None
        or page.size.width == 0
        or page.size.height == 0
    ):
        raise HTTPException(500, "Page size data is missing or invalid")
    img_w, img_h = img.size
    sx = img_w / page.size.width
    sy = img_h / page.size.height

    draw = ImageDraw.Draw(img)
    PAD = 6
    LINE_WIDTH = 4
    COLORS = ["#FF4136", "#0074D9", "#2ECC40", "#FFDC00", "#B10DC9"]
    color_cycle = cycle(COLORS)
    inset = PAD + LINE_WIDTH

    for box in req.boxes:
        color = next(color_cycle)
        l_f = box.l * sx - inset
        t_f = img_h - box.t * sy - inset
        r_f = box.r * sx + inset
        b_f = img_h - box.b * sy + inset

        left = math.floor(l_f)
        top = math.floor(t_f)
        right = math.ceil(r_f)
        bottom = math.ceil(b_f)

        draw.rectangle([(left, top), (right, bottom)], outline=color, width=LINE_WIDTH)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
