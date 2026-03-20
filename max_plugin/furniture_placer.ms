/*
furniture_placer.ms  –  LayoutGPT Furniture Placer
===================================================
3ds Max 2024-2026  |  MAXScript

QUICK START
-----------
  Scripting  →  Run Script  →  furniture_placer.ms

The dialog opens.  No terminal, no environment variables, no pre-installs needed.
Python dependencies install automatically the first time.

WORKFLOW
--------
  1.  SETUP panel   – choose Local (Ollama) or Cloud (OpenAI / Custom).
  2.  ROOMS panel   – select a closed spline → click "Add Selected Room".
  3.  FURNITURE panel – select furniture objects → click "Add Selected".
                        Click a list item to edit its category, room, or copies.
  4.  GENERATE panel – click "Generate Furniture Layout".
*/

-------------------------------------------------------------------------------
-- Plugin root  (resolved from script path so it works from any location)
-------------------------------------------------------------------------------
global gFP_Dir
gFP_Dir = getFilenamePath (getThisScriptFilename())

-------------------------------------------------------------------------------
-- Global state  (persistent across rollout open/close within a session)
-------------------------------------------------------------------------------
global gFP_Rooms     -- array of structs
global gFP_Furniture -- array of structs
global gFP_Status

if gFP_Rooms == undefined     do gFP_Rooms     = #()
if gFP_Furniture == undefined do gFP_Furniture = #()
if gFP_Status == undefined    do gFP_Status    = "Ready."

-- Simple structs  -------------------------------------------------------
struct FP_Room (nodeName, roomType)
struct FP_Furn (nodeName, category, roomAssign, maxCopies, rotOff)

-- Rebuild any legacy FP_Furn instances, carrying forward fields that exist
-- and defaulting any new ones (rotOff) to 0 for backwards compatibility.
fn fp_MigrateFurniture = (
    local updated = #()
    for f in gFP_Furniture do (
        local ro = try (f.rotOff) catch (0)
        append updated (FP_Furn nodeName:f.nodeName category:f.category \
                                roomAssign:f.roomAssign maxCopies:f.maxCopies \
                                rotOff:ro)
    )
    gFP_Furniture = updated
)
fp_MigrateFurniture()

-- All LayoutGPT categories for the category dropdown
global gFP_AllCats
gFP_AllCats = #(
    "armchair","cabinet","ceiling_lamp","chair","children_cabinet",
    "coffee_table","console_table","corner_side_table","desk",
    "dining_chair","dining_table","double_bed","dresser","dressing_chair",
    "dressing_table","floor_lamp","kids_bed","l_shaped_sofa","lounge_chair",
    "loveseat_sofa","multi_seat_sofa","nightstand","pendant_lamp",
    "round_end_table","shelf","single_bed","stool","table","tv_stand",
    "wardrobe","wine_cabinet"
)

-- Python initialisation  (auto-installs openai package if missing) ------

-- Convert gFP_Dir to a Python-safe path: forward slashes, no trailing sep
fn fp_PyPath = (
    local p = substituteString gFP_Dir "\\" "/"
    if p.count > 0 and p[p.count] == "/" do p = substring p 1 (p.count - 1)
    p
)

fn fp_InitPython = (
    local pyInit = (
        "import sys, os\n" +
        "sys.path.insert(0, '" + fp_PyPath() + "')\n" +
        "from installer import ensure_dependencies\n" +
        "_ok, _msg = ensure_dependencies()\n" +
        "print('[FP]', _msg)\n"
    )
    try ( python.execute pyInit )
    catch ( print ("[FP] Python init error: " + getCurrentException()) )
)

-- Safe python.execute wrapper: always returns a string.
-- python.execute returns the MAXScript value #success (not a string) when
-- the Python code runs without error but produces no print output.
-- Using #success with findString or string concatenation throws an exception.
fn fp_PyRun code default:"" = (
    local result = default
    try (
        local r = python.execute code
        if classOf r == String then result = trimLeft (trimRight r)
    ) catch ()
    result
)

fn fp_RefreshRoomList lb = (
    local items = #()
    for r in gFP_Rooms do append items ("[" + r.roomType + "]  " + r.nodeName)
    lb.items = items
)

fn fp_RefreshFurnList lb = (
    local items = #()
    for f in gFP_Furniture do (
        local dupe = if f.maxCopies > 1 then ("  x" + f.maxCopies as string) else ""
        local rm   = if f.roomAssign != "all" then ("  [" + f.roomAssign + "]") else ""
        append items (f.category + dupe + "  —  " + f.nodeName + rm)
    )
    lb.items = items
)

fn fp_RoomNames = (
    local names = #("All Rooms")
    for r in gFP_Rooms do append names r.nodeName
    names
)

-- Serialize registry to JSON for Python  --------------------------------
fn fp_RegistryJSON = (
    local s = "{\"rooms\":["
    for i = 1 to gFP_Rooms.count do (
        local r = gFP_Rooms[i]
        local comma = if i < gFP_Rooms.count then "," else ""
        s += "{\"node_name\":\"" + r.nodeName + "\",\"room_type\":\"" + r.roomType + "\"}" + comma
    )
    s += "],\"furniture\":["
    for i = 1 to gFP_Furniture.count do (
        local f = gFP_Furniture[i]
        local comma = if i < gFP_Furniture.count then "," else ""
        s += "{\"node_name\":\"" + f.nodeName + "\",\"category\":\"" + f.category + "\","
        s += "\"room_assignment\":\"" + f.roomAssign + "\",\"max_instances\":" + (f.maxCopies as string) + ","
        s += "\"rotation_offset\":" + (f.rotOff as string) + "}" + comma
    )
    s += "]}"
    s
)

-------------------------------------------------------------------------------
-- ROLLOUT 1 – LLM Setup
-------------------------------------------------------------------------------
rollout ro_Setup "1 · LLM Setup" width:420
(
    group "Backend"
    (
        radiobuttons rdo_backend labels:#("Local – Ollama (free, no key)", "Cloud – OpenAI", "Custom endpoint") default:1 columns:1
    )

    -- Ollama section
    group "Ollama  (runs on your PC – no internet needed)"
    (
        label       lbl_olStatus  "Status: checking…" align:#left
        dropdownlist ddl_olModel  "Model:" items:#("llama3.2:3b  – fast, ~2 GB", "mistral:7b  – balanced, ~4 GB", "qwen2.5:7b  – best quality, ~4 GB") selection:1 width:280 align:#left
        button      btn_olStart   "Start Ollama"  width:130 align:#left across:2
        button      btn_olPull    "Download Model" width:130 align:#right
        button      btn_olGet     "Get Ollama (opens browser)" width:260 align:#left
        progressbar prg_pull      value:0 height:10 width:400 align:#left color:green
        label       lbl_pullMsg   "" align:#left
    )

    -- OpenAI section
    group "OpenAI"
    (
        label       lbl_oaiNote "API key is stored in memory only – never saved to disk." align:#left
        edittext    et_oaiKey   "API Key:" fieldWidth:300 align:#left
        dropdownlist ddl_oaiModel "Model:" items:#("gpt-4o","gpt-4-turbo","gpt-4","gpt-3.5-turbo") selection:1 width:200 align:#left
    )

    -- Custom section
    group "Custom  (LM Studio, vLLM, Azure, …)"
    (
        edittext    et_custURL   "Endpoint URL:"  text:"http://localhost:1234/v1" fieldWidth:300 align:#left
        edittext    et_custKey   "API Key:"        fieldWidth:300 align:#left
        edittext    et_custModel "Model name:"     fieldWidth:200 align:#left
    )

    -- Show/hide sections on backend change
    on ro_Setup open do (
        fp_InitPython()
        -- Check Ollama status
        local chk = "import sys; sys.path.insert(0,'" + fp_PyPath() + "')\n" + \
                    "from installer import ollama_status\n" + \
                    "running,installed = ollama_status()\n" + \
                    "print('RUNNING' if running else ('INSTALLED' if installed else 'MISSING'))\n"
        local result = fp_PyRun chk
        if findString result "RUNNING"   != undefined then lbl_olStatus.text = "Status: ✓ Running"
        else if findString result "INSTALLED" != undefined then lbl_olStatus.text = "Status: Installed (not running)"
        else lbl_olStatus.text = "Status: Not installed"

        -- Pre-fill OpenAI key from env if set
        -- Use dotNet for reliable env var reading across Max 2024-2026
        try (
            local sysKey = (dotNetClass "System.Environment").GetEnvironmentVariable "OPENAI_API_KEY"
            if sysKey != undefined then (
                local sysKeyStr = sysKey.ToString()
                if sysKeyStr != "" do et_oaiKey.text = sysKeyStr
            )
        ) catch ()
    )

    on btn_olStart pressed do (
        lbl_olStatus.text = "Starting Ollama…"
        windows.processPostedMessages()
        local code = "import sys; sys.path.insert(0,'" + fp_PyPath() + "')\n" + \
                     "from installer import start_ollama\n" + \
                     "ok,msg = start_ollama()\n" + \
                     "print(msg)\n"
        local msg = fp_PyRun code default:"(no response)"
        lbl_olStatus.text = "Status: " + msg
    )

    on btn_olPull pressed do (
        local modelTags = #("llama3.2:3b","mistral:7b","qwen2.5:7b")
        local tag = modelTags[ddl_olModel.selection]
        lbl_pullMsg.text = "Downloading " + tag + " …  (may take several minutes)"
        prg_pull.value = 5
        windows.processPostedMessages()
        local code = "import sys; sys.path.insert(0,'" + fp_PyPath() + "')\n" + \
                     "try:\n" + \
                     "    from installer import pull_model\n" + \
                     "    ok,msg = pull_model('" + tag + "')\n" + \
                     "    print(msg)\n" + \
                     "except Exception as _e:\n" + \
                     "    print('Error: ' + str(_e))\n"
        local msg = fp_PyRun code default:"(no response – check Ollama is installed)"
        prg_pull.value = 100
        -- Append where Ollama stores models so the user can verify
        local ollamaModelsPath = (dotNetClass "System.Environment").GetEnvironmentVariable "USERPROFILE"
        local pathStr = ""
        try ( pathStr = ollamaModelsPath.ToString() ) catch ()
        if pathStr != "" then
            lbl_pullMsg.text = msg + "  |  Models stored at: " + pathStr + "\\.ollama\\models"
        else
            lbl_pullMsg.text = msg
    )

    on btn_olGet pressed do (
        local code = "from installer import open_ollama_download_page; open_ollama_download_page()\n"
        try ( python.execute code ) catch ()
    )
)

-------------------------------------------------------------------------------
-- ROLLOUT 2 – Rooms
-------------------------------------------------------------------------------
rollout ro_Rooms "2 · Room Boundaries" width:420
(
    label lbl_hint "Select a closed spline in the viewport, then click Add." align:#left

    listbox lb_rooms ""  height:5 width:395 align:#center
    button  btn_addRoom  "Add Selected Room"  width:185 align:#left  across:3
    button  btn_remRoom  "Remove"             width:90  align:#center
    button  btn_clrRoom  "Clear All"          width:90  align:#right

    group "Selected Room"
    (
        label           lbl_rName  "—" align:#left
        label           lbl_rType  "Room type:" align:#left
        dropdownlist    ddl_rType  items:#("bedroom","livingroom") selection:1 width:160 align:#left
    )

    -- helpers
    fn _roomTypeIdx t = (
        if t == "bedroom" then 1 else 2
    )
    fn _updateDetail = (
        local sel = lb_rooms.selection
        if sel == 0 or sel > gFP_Rooms.count do (
            lbl_rName.text = "—"
            return()
        )
        local r = gFP_Rooms[sel]
        lbl_rName.text  = r.nodeName
        ddl_rType.selection = _roomTypeIdx r.roomType
    )

    on ro_Rooms open do (
        fp_RefreshRoomList lb_rooms
        _updateDetail()
    )

    on btn_addRoom pressed do (
        local added = 0
        for obj in selection do (
            -- Auto-detect room type from name; default to bedroom
            local nm   = obj.name as string
            local rType = "bedroom"
            local nmL  = toLower nm
            if (findString nmL "living") != undefined or (findString nmL "lounge") != undefined do rType = "livingroom"
            if (findString nmL "dining") != undefined do rType = "livingroom"

            -- Avoid duplicates
            local dup = false
            for r in gFP_Rooms where r.nodeName == nm do dup = true
            if not dup do (
                append gFP_Rooms (FP_Room nodeName:nm roomType:rType)
                added += 1
            )
        )
        fp_RefreshRoomList lb_rooms
        if added == 0 do messageBox "Nothing new was added.\nTip: select a spline first." title:"Add Room"
    )

    on btn_remRoom pressed do (
        local sel = lb_rooms.selection
        if sel > 0 and sel <= gFP_Rooms.count do (
            deleteItem gFP_Rooms sel
            fp_RefreshRoomList lb_rooms
            _updateDetail()
        )
    )

    on btn_clrRoom pressed do (
        if queryBox "Remove all rooms?" title:"Confirm" do (
            gFP_Rooms = #()
            fp_RefreshRoomList lb_rooms
            _updateDetail()
        )
    )

    on lb_rooms selected i do _updateDetail()

    on ddl_rType selected i do (
        local sel = lb_rooms.selection
        if sel > 0 and sel <= gFP_Rooms.count do (
            gFP_Rooms[sel].roomType = ddl_rType.items[i]
        )
    )
)

-------------------------------------------------------------------------------
-- ROLLOUT 3 – Furniture
-------------------------------------------------------------------------------
rollout ro_Furniture "3 · Furniture Assets" width:420
(
    label lbl_hint2 "Select furniture objects in the viewport, then click Add." align:#left

    listbox lb_furn "" height:7 width:395 align:#center
    button  btn_addFurn "Add Selected"  width:120 align:#left  across:3
    button  btn_remFurn "Remove"        width:90  align:#center
    button  btn_clrFurn "Clear All"     width:90  align:#right

    group "Selected Item"
    (
        label           lbl_fName       "—" align:#left

        label           lbl_cat         "Category:" align:#left
        dropdownlist    ddl_cat         items:#() selection:1 width:220 align:#left

        label           lbl_room        "Appears in room:" align:#left
        dropdownlist    ddl_room        items:#("All Rooms") selection:1 width:220 align:#left

        label           lbl_copies      "Max copies placed:" align:#left
        spinner         spn_copies      "" range:[1,20,1] type:#integer fieldWidth:50 align:#left
        label           lbl_copiesHint  "(1 = no duplicates)" align:#left

        label           lbl_rotOff      "Rotation offset (°):" align:#left
        spinner         spn_rotOff      "" range:[-360,360,0] type:#integer fieldWidth:60 align:#left
        label           lbl_rotOffHint  "(use 180 if asset faces backwards)" align:#left

        button          btn_applyItem   "Apply Changes" width:140 align:#left
    )

    fn _furnitureDetail = (
        local sel = lb_furn.selection
        if sel == 0 or sel > gFP_Furniture.count do (
            lbl_fName.text = "—"
            return()
        )
        local f = gFP_Furniture[sel]
        lbl_fName.text = f.nodeName

        -- Set category dropdown
        local catIdx = 1
        for i = 1 to gFP_AllCats.count where gFP_AllCats[i] == f.category do catIdx = i
        ddl_cat.selection = catIdx

        -- Set room dropdown
        local roomItems = fp_RoomNames()
        ddl_room.items = roomItems
        local rIdx = 1
        for i = 1 to roomItems.count where roomItems[i] == f.roomAssign do rIdx = i
        ddl_room.selection = rIdx

        spn_copies.value    = f.maxCopies
        spn_rotOff.value    = f.rotOff
    )

    on ro_Furniture open do (
        -- Migrate any legacy FP_Furn structs that pre-date the baseRotOffset
        -- field before touching the list (prevents "Unknown property" crash).
        fp_MigrateFurniture()
        ddl_cat.items = gFP_AllCats
        fp_RefreshFurnList lb_furn
        _furnitureDetail()
    )

    on btn_addFurn pressed do (
        local added = 0
        for obj in selection do (
            local nm = obj.name as string
            local dup = false
            for f in gFP_Furniture where f.nodeName == nm do dup = true
            if not dup do (
                -- Auto-detect category via Python tag_mapper
                local catCode = (
                    "import sys; sys.path.insert(0, '" + fp_PyPath() + "')\n" +
                    "from layoutgpt_bridge.tag_mapper import normalize_tag\n" +
                    "cat = normalize_tag('" + nm + "') or 'armchair'\n" +
                    "print(cat)\n"
                )
                local cat = fp_PyRun catCode default:"armchair"
                append gFP_Furniture (FP_Furn nodeName:nm category:cat roomAssign:"all" maxCopies:1 rotOff:0)
                added += 1
            )
        )
        fp_RefreshFurnList lb_furn
        if added == 0 do messageBox "Nothing new was added.\nTip: select furniture objects first." title:"Add Furniture"
        -- Refresh room dropdown in detail panel
        ddl_room.items = fp_RoomNames()
    )

    on btn_remFurn pressed do (
        local sel = lb_furn.selection
        if sel > 0 and sel <= gFP_Furniture.count do (
            deleteItem gFP_Furniture sel
            fp_RefreshFurnList lb_furn
            _furnitureDetail()
        )
    )

    on btn_clrFurn pressed do (
        if queryBox "Remove all furniture?" title:"Confirm" do (
            gFP_Furniture = #()
            fp_RefreshFurnList lb_furn
            _furnitureDetail()
        )
    )

    on lb_furn selected i do (
        ddl_cat.items  = gFP_AllCats
        ddl_room.items = fp_RoomNames()
        _furnitureDetail()
    )

    on btn_applyItem pressed do (
        local sel = lb_furn.selection
        if sel > 0 and sel <= gFP_Furniture.count do (
            gFP_Furniture[sel].category     = gFP_AllCats[ddl_cat.selection]
            local roomItems = fp_RoomNames()
            local rSel      = ddl_room.selection
            gFP_Furniture[sel].roomAssign   = if rSel == 1 then "all" else roomItems[rSel]
            gFP_Furniture[sel].maxCopies    = spn_copies.value as integer
            gFP_Furniture[sel].rotOff       = spn_rotOff.value as integer
            fp_RefreshFurnList lb_furn
        )
    )
)

-------------------------------------------------------------------------------
-- ROLLOUT 4 – Generate
-------------------------------------------------------------------------------
rollout ro_Generate "4 · Generate" width:420
(
    group "Options"
    (
        checkbox    chk_snap    "Snap furniture to room floor"   checked:true
        checkbox    chk_scale   "Scale instances to LLM dimensions" checked:false
        label       lbl_temp    "Creativity (temperature):" align:#left
        spinner     spn_temp    "" range:[0.0,2.0,0.7] type:#float fieldWidth:55 align:#left
    )

    button  btn_gen     "Generate Furniture Layout"  width:380 height:40 align:#center
    button  btn_clear   "Clear Last Layout"          width:380 height:28 align:#center

    label   lbl_status  "Ready." align:#left
    edittext et_log     "" height:80 width:395 readOnly:true align:#center

    fn _resolveBackend = (
        -- Returns (backend, apiKey, model, baseUrl)
        local backend  = "openai"
        local apiKey   = ""
        local model    = "gpt-4o"
        local baseUrl  = ""

        try (
            local sel = ro_Setup.rdo_backend.state
            if sel == 1 do (
                backend = "ollama"
                local tags = #("llama3.2:3b","mistral:7b","qwen2.5:7b")
                model   = tags[ro_Setup.ddl_olModel.selection]
                apiKey  = "ollama"
                baseUrl = "http://localhost:11434/v1"
            )
            if sel == 2 do (
                backend = "openai"
                apiKey  = ro_Setup.et_oaiKey.text
                local ms = #("gpt-4o","gpt-4-turbo","gpt-4","gpt-3.5-turbo")
                model   = ms[ro_Setup.ddl_oaiModel.selection]
                baseUrl = ""
            )
            if sel == 3 do (
                backend = "custom"
                baseUrl = ro_Setup.et_custURL.text
                apiKey  = ro_Setup.et_custKey.text
                model   = ro_Setup.et_custModel.text
            )
        ) catch ()
        #(backend, apiKey, model, baseUrl)
    )

    on btn_gen pressed do (
        lbl_status.text = "Running… (see MAXScript Listener for live output)"
        et_log.text     = ""
        windows.processPostedMessages()

        local params = _resolveBackend()
        local backend = params[1]
        local apiKey  = params[2]
        local model   = params[3]
        local baseUrl = params[4]
        local regJSON = fp_RegistryJSON()

        -- Escape backslashes and quotes for embedding in Python string
        local regEsc  = substituteString regJSON "\\" "\\\\"
        regEsc        = substituteString regEsc  "\"" "\\\""

        local baseUrlLine = if baseUrl != "" \
            then ("    'base_url'       : '" + baseUrl + "',\n") \
            else ("    'base_url'       : None,\n")

        local pyCode = (
            "import sys, os\n" +
            "sys.path.insert(0, '" + fp_PyPath() + "')\n" +
            -- Clear cached modules so code changes on disk are picked up every run.
            -- Without this, 3ds Max's Python interpreter reuses the first-import
            -- snapshot for the entire session, ignoring any file edits.
            "_stale = [k for k in list(sys.modules.keys())\n" +
            "          if k == 'furniture_placer' or k.startswith('layoutgpt_bridge')\n" +
            "          or k == 'installer']\n" +
            "for _m in _stale: del sys.modules[_m]\n" +
            "import furniture_placer as fp\n" +
            "cfg = {\n" +
            "    'backend'         : '" + backend + "',\n" +
            "    'api_key'         : '" + apiKey + "',\n" +
            "    'model'           : '" + model + "',\n" +
            baseUrlLine +
            "    'temperature'     : " + ((ro_Generate.spn_temp.value) as string) + ",\n" +
            "    'snap_to_floor'   : " + (if chk_snap.checked  then "True" else "False") + ",\n" +
            "    'scale_to_fit'    : " + (if chk_scale.checked then "True" else "False") + ",\n" +
            "    'registry_json'   : \"" + regEsc + "\",\n" +
            "}\n" +
            "result = fp.main(cfg)\n" +
            "print(result)\n"
        )

        local result = ""
        try (
            local r = python.execute pyCode
            result = if classOf r == String then r else "Done."
            lbl_status.text = "Done."
        ) catch (
            result = "Python error: " + getCurrentException()
            lbl_status.text = "Error — see log below."
        )
        et_log.text = result
    )

    on btn_clear pressed do (
        local toDelete = #()
        for obj in objects where (findString obj.name "_layout") != undefined do
            append toDelete obj
        if toDelete.count > 0 then (
            if queryBox ("Delete " + (toDelete.count as string) + " layout object(s)?") title:"Confirm" do (
                delete toDelete
                lbl_status.text = "Cleared " + (toDelete.count as string) + " object(s)."
            )
        ) else (
            lbl_status.text = "No _layout objects found."
        )
    )
)

-------------------------------------------------------------------------------
-- Assemble the floater
-------------------------------------------------------------------------------
if (fp_Floater != undefined) do ( try (closeRolloutFloater fp_Floater) catch() )
global fp_Floater
fp_Floater = newRolloutFloater "LayoutGPT Furniture Placer  v2" 440 820

addRollout ro_Setup     fp_Floater
addRollout ro_Rooms     fp_Floater
addRollout ro_Furniture fp_Floater
addRollout ro_Generate  fp_Floater
