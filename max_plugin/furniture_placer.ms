/*
furniture_placer.ms
-------------------
MAXScript UI wrapper for the LayoutGPT Furniture Placer plugin.
Compatible with 3ds Max 2024-2026.

Usage
-----
  1. Open the MAXScript Editor (Scripting → MAXScript Editor).
  2. File → Open → furniture_placer.ms
  3. Click Evaluate All  (Ctrl+E), then close the editor.
  4. A floating dialog will appear.  Fill in settings and click "Place Furniture".

The script delegates all heavy lifting to furniture_placer.py via pymxs / Python.
*/

-- ============================================================
-- Helper: resolve plugin directory from this script's path
-- ============================================================
global gFP_PluginDir
gFP_PluginDir = getFilenamePath (getThisScriptFilename())

-- ============================================================
-- UI definition
-- ============================================================
rollout ro_FurniturePlacer "LayoutGPT Furniture Placer" width:400 height:480
(
    -- ---- Header ----
    label lbl_title "LayoutGPT Furniture Placer" align:#center
    label lbl_sub   "Auto-furnish closed spline boundaries" align:#center
    separator sep1

    -- ---- OpenAI settings ----
    group "OpenAI Settings"
    (
        label       lbl_key     "API Key:" align:#left
        edittext    et_apiKey   "" width:360 height:22 align:#center \
                                tooltip:"OpenAI API key (or set OPENAI_API_KEY env var)"
        label       lbl_model   "Model:" align:#left
        dropdownlist ddl_model  items:#("gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo") \
                                selection:1 width:200 align:#left
        label       lbl_temp    "Temperature:" align:#left
        spinner     spn_temp    "" range:[0.0, 2.0, 0.7] type:#float fieldwidth:60 align:#left
    )

    separator sep2

    -- ---- Dataset stats ----
    group "LayoutGPT Dataset (optional)"
    (
        label       lbl_ds      "ATISS data_output dir:" align:#left
        edittext    et_dsDir    "" width:320 height:22 align:#left \
                                tooltip:"e.g. C:\\LayoutGPT\\ATISS\\data_output  (leave blank to skip)"
        button      btn_dsDir   "Browse…" width:60 align:#right across:2
        edittext    et_dsDir2   ""  -- dummy for layout alignment; hidden
    )

    separator sep3

    -- ---- Placement options ----
    group "Placement Options"
    (
        checkbox    chk_scale   "Scale instances to LLM bounding box" checked:false
        checkbox    chk_snap    "Snap instances to room floor Z"       checked:true
        label       lbl_nlay    "Layout variants per room:" align:#left
        spinner     spn_nlay    "" range:[1, 5, 1] type:#integer fieldwidth:50 align:#left
    )

    separator sep4

    -- ---- Tag prefix overrides ----
    group "Tag Prefixes (comma-separated)"
    (
        label       lbl_fpfx    "Furniture prefix(es):" align:#left
        edittext    et_fpfx     "furn_, furniture_" width:360 align:#center
        label       lbl_rpfx    "Room prefix(es):" align:#left
        edittext    et_rpfx     "room_, boundary_, area_" width:360 align:#center
    )

    separator sep5

    -- ---- Action buttons ----
    button btn_run    "Place Furniture"   width:180 height:36 align:#left across:2
    button btn_clear  "Clear Last Layout" width:180 height:36 align:#right

    -- ---- Status ----
    label lbl_status "Ready." align:#left

    -- ================================================================
    -- Event handlers
    -- ================================================================

    -- Browse for dataset dir
    on btn_dsDir pressed do
    (
        local d = getSavePath caption:"Select ATISS data_output folder" \
                              initialDir:(if et_dsDir.text != "" then et_dsDir.text else "C:\\")
        if d != undefined do et_dsDir.text = d
    )

    -- Main "Place Furniture" action
    on btn_run pressed do
    (
        lbl_status.text = "Running…"
        windows.processPostedMessages()

        try
        (
            -- Build config dict as a Python string so we can pass it to exec
            local apiKey   = et_apiKey.text
            local model    = ddl_model.items[ddl_model.selection]
            local temp     = spn_temp.value as float
            local dsDir    = et_dsDir.text
            local scaleFit = if chk_scale.checked then "True" else "False"
            local snapFlr  = if chk_snap.checked  then "True" else "False"
            local nLayouts = spn_nlay.value as integer

            -- Parse prefix lists
            local fPfx = "(" + \
                (substituteString \
                    ("\"" + (substituteString et_fpfx.text ", " "\", \"") + "\"") \
                    "\"\"" "\"furn_\"") + \
                ",)"
            local rPfx = "(" + \
                (substituteString \
                    ("\"" + (substituteString et_rpfx.text ", " "\", \"") + "\"") \
                    "\"\"" "\"room_\"") + \
                ",)"

            -- Build Python config and call main()
            local pyCode = (
                "import sys, os\n" +
                "sys.path.insert(0, r'" + gFP_PluginDir + "')\n" +
                "import furniture_placer as fp\n" +
                "cfg = {\n" +
                "    'openai_api_key' : '" + apiKey + "',\n" +
                "    'openai_model'   : '" + model + "',\n" +
                "    'temperature'    : " + (temp as string) + ",\n" +
                "    'dataset_dir'    : " + (if dsDir == "" then "None" else ("r'" + dsDir + "'")) + ",\n" +
                "    'scale_to_fit'   : " + scaleFit + ",\n" +
                "    'snap_to_floor'  : " + snapFlr + ",\n" +
                "    'n_layouts'      : " + (nLayouts as string) + ",\n" +
                "    'furn_prefixes'  : " + fPfx + ",\n" +
                "    'room_prefixes'  : " + rPfx + ",\n" +
                "}\n" +
                "fp.main(cfg)\n"
            )

            python.execute pyCode
            lbl_status.text = "Done. Check the Listener for details."
        )
        catch
        (
            local msg = getCurrentException()
            lbl_status.text = "Error: " + msg
            messageBox ("Furniture Placer error:\n" + msg) title:"Error" beep:true
        )
    )

    -- Clear last layout (deletes all objects whose names contain "_layout")
    on btn_clear pressed do
    (
        local toDelete = #()
        for obj in objects do
        (
            if (findString obj.name "_layout") != undefined do
                append toDelete obj
        )
        if toDelete.count > 0 then
        (
            local ans = queryBox ("Delete " + (toDelete.count as string) +
                                  " layout object(s)?") title:"Confirm"
            if ans do delete toDelete
            lbl_status.text = "Cleared " + (toDelete.count as string) + " object(s)."
        )
        else
            lbl_status.text = "Nothing to clear."
    )

    -- Populate API key from environment on open
    on ro_FurniturePlacer open do
    (
        local envKey = python.execute "import os; print(os.environ.get('OPENAI_API_KEY',''))"
        -- python.execute returns a string only from a print; try systemEnvironmentVariable too
        local sysKey = systemEnvironmentVariable "OPENAI_API_KEY"
        if sysKey != undefined and sysKey != "" do
            et_apiKey.text = sysKey
    )
)

-- ============================================================
-- Show the dialog
-- ============================================================
if (roFloater != undefined) do closeRolloutFloater roFloater
roFloater = newRolloutFloater "LayoutGPT Furniture Placer" 420 510
addRollout ro_FurniturePlacer roFloater
