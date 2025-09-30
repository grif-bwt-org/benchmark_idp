PROMPT_TABLE_EXTRACTION = """
Title: Comprehensive Extraction of All Data from Engineering Drawing into a Detailed JSON Structure
Prompt:
Act as a meticulous Quality Control Engineer and Metrology expert. Your task is to perform a complete and exhaustive analysis of the provided multi-sheet engineering blueprint and extract ALL specified information into a single, highly structured JSON object.
The output must be a single, well-formed JSON object.
The top-level keys of this JSON object should be strings representing the names of the different views, sections, and details from the drawing. Crucially, include the scale and any applicability notes in the view name (e.g., "Section B-B (2.5:1) (2 places)", "Detail M (20:1)").
The value for each view key must be an array of objects. Each object in the array represents a single, complete dimensional callout or specification found within that view.
Each dimension object must adhere to the following detailed schema:
* id (string or null): The identifier from the drawing, if present (e.g., "DIM44"). Use null if not present.
* dimension_type (string): The type of dimension. Must be one of: linear, diameter, radius, angle, chamfer, thread, surface_roughness.
* nominal_value (number or string): The primary numerical value of the dimension (e.g., 17, 3.2, M3).
* feature_description (string or null): A brief text description of the feature (e.g., "holes", "slots", "septum surface").
* feature_count (integer or null): The number of identical features this dimension applies to (e.g., for "8 отв.", this would be 8). If not specified, use null.
* upper_tolerance (number or null): The numerical upper tolerance value (e.g., 0.02).
* lower_tolerance (number or null): The numerical lower tolerance value (e.g., -0.02).
* tolerance_class (string or null): The alphanumeric tolerance class (e.g., "H7", "F7", "6H").
* is_reference (boolean): Set to true if the dimension is marked as a reference dimension (typically with an asterisk *), false otherwise.
* notes (string or null): Any additional notes directly associated with the dimension, such as thread depth (e.g., "depth 7.5-8mm").
* geometric_tolerances (array of objects or null): An array to capture all geometric dimensioning and tolerancing (GD&T) frames associated with this feature. Each object in the array should have:
    * type (string): The type of geometric control (e.g., position, perpendicularity, parallelism, flatness, total_runout).
    * value (number): The tolerance value (e.g., 0.05).
    * datums (array of strings): An array of the datum references (e.g., ["Д", "Г"], ["M"]).
    * zone_modifier (string or null): Any material condition modifier on the tolerance value, if present (e.g., M for Maximum Material Condition).
Additional Instructions:
1. Be Exhaustive: Do not skip any information. Every number, symbol, and note on the drawing is important.
2. View Naming: Be precise with view names. For the main, unlabelled views, use standard names like "Main Front View", "Main Top View". For labeled views, use their labels and add scale/context, like "Section A-A", "View B (2.5:1)".
3. General Notes & Title Block: Create two separate top-level keys:
    * general_notes: An array of strings containing all numbered or general notes from the drawing. Translate key technical terms if possible but preserve the original text.
    * title_block: An object containing key information from the title block, such as part_number, part_name, material, mass, and scale.
4. Surface Finish: Global surface finish symbols (like Ra 3.2 (✓)) should be captured in general_notes. Local surface finish symbols should be treated as a feature within their respective view.
Example of desired JSON structure based on the provided complex drawing:
Generated json

{
  "title_block": {
    "part_name": "Вед. инж.",
    "material": "АМг6 ГОСТ 4784-2019",
    "mass": "101.39 г",
    "scale": "2:1",
    "part_number": null
  },
  "Main Front View": [
    {
      "id": "DIM11",
      "dimension_type": "diameter",
      "nominal_value": 3.2,
      "feature_description": "holes",
      "feature_count": 8,
      "upper_tolerance": null,
      "lower_tolerance": null,
      "tolerance_class": null,
      "is_reference": false,
      "notes": null,
      "geometric_tolerances": null
    },
    {
      "id": "DIM7",
      "dimension_type": "diameter",
      "nominal_value": 1.5,
      "feature_description": "holes",
      "feature_count": 2,
      "upper_tolerance": 0.015,
      "lower_tolerance": 0.006,
      "tolerance_class": "F7",
      "is_reference": false,
      "notes": null,
      "geometric_tolerances": [
        {
          "type": "position",
          "value": 0.02,
          "datums": ["Д", "Л"],
          "zone_modifier": "M" 
        }
      ]
    },
    {
      "id": "DIM4",
      "dimension_type": "linear",
      "nominal_value": 5.8,
      "feature_description": null,
      "feature_count": null,
      "upper_tolerance": 0.018,
      "lower_tolerance": 0,
      "tolerance_class": null,
      "is_reference": true,
      "notes": null,
      "geometric_tolerances": null
    }
  ],
  "Section B-B (2.5:1) (2 places)": [
    {
      "id": "DIM27",
      "dimension_type": "thread",
      "nominal_value": "M3",
      "feature_description": "threaded holes",
      "feature_count": 4,
      "upper_tolerance": null,
      "lower_tolerance": null,
      "tolerance_class": "6H",
      "is_reference": false,
      "notes": "depth 7.5-8mm",
      "geometric_tolerances": null
    }
  ],
  "general_notes": [
    "1. * Размеры для справок.",
    "2. Общие допуски по ГОСТ 30893.1: H12, h12, ±IT12/2.",
    "3. Общие допуски формы и расположения - ГОСТ 30893.2-Н.",
    "4. Неуказанные размеры согласно CAD модели.",
    "7. Шероховатость поверхностей внутренних каналов и фланцев — Ra 1,6.",
    "8. Покрытие Хим. Н9.МЗ.Ср6.",
    "9. Допускается отсутствие покрытия в резьбовых отверстиях.",
    "10. Маркировать 02 согласно СТП-3.",
    "11. Допуск по массе согласно СТП-4.",
    "Global surface roughness: Ra 3.2"
  ]
}
"""