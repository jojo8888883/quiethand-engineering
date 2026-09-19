(() => {
  if (typeof records === "undefined") return;

  const storageKey = "quiethand-m3-v1-2-user-judgements-v1";
  const notesStorageKey = "quiethand-m3-v1-2-user-notes-v1";
  let judgements = {};
  let notes = {};
  try {
    judgements = JSON.parse(localStorage.getItem(storageKey) || "{}");
  } catch (_error) {
    judgements = {};
  }
  try {
    notes = JSON.parse(localStorage.getItem(notesStorageKey) || "{}");
  } catch (_error) {
    notes = {};
  }

  const recordById = new Map(records.map((record) => [record.event_id, record]));
  const actionZh = {
    "adjusting":"调整", "aligning":"对齐", "applying adhesive":"涂胶", "applying paint":"刷漆",
    "assembling":"组装", "beating":"敲打", "brushing":"刷洗", "cleaning":"清洁",
    "comparing":"比较", "cutting":"切割", "grasp":"抓取", "grasping":"抓取",
    "holding":"拿取或握持", "manipulating":"操作", "measuring":"测量", "moving":"移动",
    "painting":"刷漆", "placing":"放置", "pointing":"指向", "pouring":"倾倒",
    "pressing":"按压", "repositioning":"调整位置", "scooping":"舀取", "scraping":"刮擦",
    "scrubbing":"擦洗", "sharpening":"磨削", "stirring":"搅拌", "tightening a screw":"拧紧螺丝",
    "unknown":"未识别", "unscrewing":"拧松螺丝", "poking":"戳动", "erasing":"擦除",
    "aligning objects":"对齐物体", "removing a small object from the wheel":"从轮子上取下小物体",
    "pouring tea":"倒茶", "applying adhesive or coating":"涂抹胶或涂层",
    "manipulating objects with a slotted spoon":"用漏勺操作物体", "manipulating spoon":"操作勺子",
    "straining":"滤取", "lifting and rotating":"抬起并旋转",
    "assembling a wooden box with a roller tool":"用滚筒工具组装木盒",
    "pouring liquid":"倾倒液体", "using a lint roller on a kettle":"用粘毛滚筒处理水壶",
    "grinding":"打磨",
    "using a clothes steamer on a kettle":"用挂烫机处理水壶",
    "using a lint remover on a kettle":"用去毛器处理水壶", "washing":"清洗"
  };
  const objectZh = {
    "black rectangular object":"黑色长方形物体", "blue bowl":"蓝色碗", "brush":"刷子",
    "cleaver":"菜刀", "clothes roller":"粘毛滚筒", "clothes steamer":"挂烫机",
    "digital caliper":"数显卡尺", "electric cleaner":"电动清洁器", "green brush":"绿色刷子",
    "green cleaning tool":"绿色清洁工具", "knife":"刀", "ladle":"汤勺",
    "lint remover":"去毛器", "pot":"锅", "red eraser":"红色橡皮",
    "red scrubber":"红色清洁刷", "red sponge":"红色海绵", "roller":"滚筒",
    "ruler":"尺子", "scissors":"剪刀", "scraping tool":"刮具", "screwdriver":"螺丝刀",
    "scrubber":"清洁刷", "slotted spoon":"漏勺", "spatula":"锅铲", "spoon":"勺子",
    "teapot":"茶壶", "unknown":"未识别", "white brush":"白色刷子", "white handle":"白色手柄",
    "white rectangular device":"白色长方形设备", "wok":"炒锅", "wooden spatula":"木铲",
    "wooden spoon":"木勺", "yellow kettle":"黄色水壶", "yellow pitcher":"黄色壶",
    "black pot":"黑色锅", "black tray":"黑色托盘", "bowl":"碗", "ceramic cup":"陶瓷杯",
    "container":"容器", "fabric":"布料", "ice in container":"容器里的冰", "kettle":"水壶",
    "kettles":"几个水壶", "material":"材料", "metal plate":"金属板", "metallic strip":"金属条",
    "pan":"平底锅", "red bowl":"红色碗", "tablecloth":"桌布", "tea tray":"茶盘",
    "tray":"托盘", "wheel":"轮子", "white beads":"白色珠子", "white container":"白色容器",
    "white kettle":"白色水壶", "white perforated tray":"白色带孔托盘", "wooden board":"木板",
    "wooden box":"木盒", "wooden cutting board":"木砧板", "wooden object":"木制物体",
    "yellow bowl":"黄色碗", "black bowl":"黑色碗", "cream tray":"米白色托盘",
    "roller tool":"滚筒工具", "contents of the container":"容器里的内容物",
    "white pitcher":"白色壶", "black wok":"黑色炒锅", "green lint roller":"绿色粘毛滚筒",
    "lint roller":"粘毛滚筒", "grinder":"打磨工具"
  };

  function zh(map, value) {
    return map[value] || value || "未识别";
  }

  function contactedObject(record, hand) {
    if (record[hand].contact === "tool") return zh(objectZh, record.tool);
    if (record[hand].contact === "target") return zh(objectZh, record.target);
    return "物体";
  }

  function actionClause(record, actor) {
    const action = zh(actionZh, record.action);
    const tool = zh(objectZh, record.tool);
    const target = zh(objectZh, record.target);
    switch (record.action) {
      case "cutting": return `${actor}用${tool}切${target}`;
      case "measuring": return `${actor}用${tool}测量${target}`;
      case "stirring": return `${actor}用${tool}搅拌${target}`;
      case "pouring": return `${actor}拿着${tool}往${target}里倒`;
      case "scooping": return `${actor}用${tool}从${target}里舀取`;
      case "scraping": return `${actor}用${tool}刮${target}`;
      case "scrubbing":
      case "brushing":
      case "cleaning":
      case "washing": return `${actor}用${tool}清洁${target}`;
      case "applying paint":
      case "painting": return `${actor}用${tool}给${target}刷漆`;
      case "applying adhesive": return `${actor}用${tool}给${target}涂胶`;
      case "beating": return `${actor}用${tool}敲打${target}`;
      case "pressing": return `${actor}用${tool}按压${target}`;
      case "sharpening": return `${actor}用${tool}磨${target}`;
      case "tightening a screw": return `${actor}用${tool}拧紧${target}上的螺丝`;
      case "unscrewing": return `${actor}用${tool}拧松${target}上的螺丝`;
      case "pointing": return `${actor}用${tool}指向${target}`;
      case "placing": return `${actor}把${tool}放到${target}处`;
      case "moving": return `${actor}移动${tool}`;
      case "repositioning": return `${actor}重新摆放${tool}`;
      case "adjusting": return `${actor}调整${tool}和${target}的位置`;
      case "aligning": return `${actor}用${tool}对齐${target}`;
      case "assembling": return `${actor}拿着${tool}组装${target}`;
      case "comparing": return `${actor}拿着${tool}比较${target}`;
      case "grasp":
      case "grasping": return `${actor}抓住${tool}`;
      case "holding": return `${actor}拿着${tool}`;
      case "manipulating": return record.target !== "unknown" && record.target !== record.tool
        ? `${actor}用${tool}处理${target}`
        : `${actor}正在操作${tool}`;
      case "using a clothes steamer on a kettle": return `${actor}用挂烫机处理水壶`;
      case "using a lint remover on a kettle": return `${actor}用去毛器处理水壶`;
      default:
        if (record.tool !== "unknown" && record.target !== "unknown") return `${actor}拿着${tool}对${target}进行${action}`;
        if (record.tool !== "unknown") return `${actor}正在${action}${tool}`;
        return `${actor}正在进行${action}`;
    }
  }

  function naturalDescription(record) {
    const names = {left:"左手", right:"右手"};
    const active = ["left", "right"].filter((hand) => record[hand].role === "active");
    const support = ["left", "right"].filter((hand) => record[hand].role === "support");
    if (!active.length) {
      if (record.action === "unknown") {
        const knownObjects = [];
        if (record.tool !== "unknown") knownObjects.push(zh(objectZh, record.tool));
        if (record.target !== "unknown" && record.target !== record.tool) knownObjects.push(zh(objectZh, record.target));
        const seen = knownObjects.length ? `，但识别到了${knownObjects.join("和")}` : "";
        if (support.length === 2) return `模型没看清具体动作${seen}，并判断两只手都在扶着或稳定物体。`;
        if (support.length === 1) return `模型没看清具体动作${seen}，只判断出${names[support[0]]}在扶着或稳定${contactedObject(record, support[0])}。`;
        return `模型没有看清这段具体在做什么${seen}，也没有判断出两只手的分工。`;
      }
      return `模型认为这段是在${zh(actionZh, record.action)}，但没有判断出哪只手在主导动作。`;
    }

    const actor = active.length === 2 ? "两只手一起" : names[active[0]];
    const clauses = [actionClause(record, actor)];
    if (support.length === 1) clauses.push(`${names[support[0]]}扶住或稳定${contactedObject(record, support[0])}`);
    if (support.length === 2) clauses.push("两只手都在扶住或稳定物体");
    return `模型认为：${clauses.join("，")}。`;
  }

  function missingDescription(record) {
    const issues = [];
    if (record.semantic_status !== "observed") issues.push("语义描述没有成功生成");
    if (Object.values(record.raw_hand).some((item) => item.status !== "observed")) issues.push("有些画面没有识别到手");
    const segmentationMissing = Object.values(record.segmentation).some((item) => item.status !== "observed");
    if (segmentationMissing) {
      issues.push("工具或作用对象没有被完整分割，相应的三维位置也缺失");
    } else if (Object.values(record.object_state).some((item) => item.status !== "observed")) {
      issues.push("物体的三维位置没有完整识别");
    }
    return issues.length ? `这条还有缺项：${issues.join("；")}。` : "";
  }

  const style = document.createElement("style");
  style.textContent = `
    .metrics { display:none !important; }
    .model-label-title { color:#7dd3fc; font-size:12px; font-weight:700; letter-spacing:.04em; margin-bottom:5px; }
    .judgement { font-size:16px; line-height:1.7; }
    .detail:empty { display:none; }
    .detail { margin-top:8px; color:#fbbf24; }
    details { margin-top:11px; color:var(--muted); }
    details summary { cursor:pointer; color:#8aa0b3; }
    .video-shell { position:relative; background:#000; }
    .video-caption { position:absolute; z-index:2; top:0; left:0; right:0; padding:7px 12px; background:#081018ee; color:#f4f8fb; font-size:13px; line-height:1.4; pointer-events:none; }
    .user-review { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-top:13px; padding-top:12px; border-top:1px solid var(--line); }
    .user-review strong { margin-right:2px; }
    .user-review button { cursor:pointer; padding:7px 14px; }
    .user-review button[data-choice="ok"].selected { border-color:var(--good); background:#123622; color:var(--good); }
    .user-review button[data-choice="minor"].selected { border-color:#60a5fa; background:#172f50; color:#93c5fd; }
    .user-review button[data-choice="bad"].selected { border-color:var(--bad); background:#3a1620; color:#fda4af; }
    .user-review .clear { color:var(--muted); padding-inline:9px; }
    .user-note { width:100%; min-height:68px; resize:vertical; margin-top:2px; padding:10px 12px; border:1px solid #35516a; border-radius:10px; background:#0b1721; color:var(--text); font:14px/1.5 inherit; }
    .archive-button { cursor:pointer; border-color:#3b82a6; }
    .review-counts { margin-left:auto; align-self:center; color:var(--muted); white-space:nowrap; }
    .card.user-ok { box-shadow:0 0 0 2px #4ade8055,0 12px 30px #0005; }
    .card.user-minor { box-shadow:0 0 0 2px #60a5fa66,0 12px 30px #0005; }
    .card.user-bad { box-shadow:0 0 0 2px #fb718566,0 12px 30px #0005; }
  `;
  document.head.appendChild(style);

  document.querySelector("h1").textContent = "QuietHand：模型看懂了什么";
  document.querySelector(".lead").textContent = "每条结果都改成了中文自然语言。先看视频，再看模型说得对不对。";
  document.querySelector(".legend").innerHTML = '<span class="swatch" style="background:#ff8228"></span>主要操作物体 <span class="swatch" style="background:#28beff"></span>作用对象 <span class="swatch" style="background:#32aaff"></span>左手 <span class="swatch" style="background:#ffd23c"></span>右手';
  document.querySelector(".limit").textContent = "可以原样使用就点“OK”；核心正确、只有颜色或措辞等无伤大雅的问题就点“有小问题”并写备注；核心动作、物体、双手分工或几何结果错误则点“不行”。";
  const filterNames = {complete: "信息齐全", partial: "部分没识别", failed: "没有结果"};
  document.querySelectorAll("button[data-filter]").forEach((button) => {
    if (filterNames[button.dataset.filter]) button.textContent = filterNames[button.dataset.filter];
  });

  const toolbar = document.querySelector(".toolbar");
  const archiveButton = document.createElement("button");
  archiveButton.type = "button";
  archiveButton.className = "archive-button";
  archiveButton.textContent = "导出标注存档";
  toolbar.appendChild(archiveButton);
  const counts = document.createElement("div");
  counts.className = "review-counts";
  toolbar.appendChild(counts);

  function save() {
    localStorage.setItem(storageKey, JSON.stringify(judgements));
  }

  function saveNotes() {
    localStorage.setItem(notesStorageKey, JSON.stringify(notes));
  }

  function updateCounts() {
    const ok = records.filter((record) => judgements[record.event_id] === "ok").length;
    const minor = records.filter((record) => judgements[record.event_id] === "minor").length;
    const bad = records.filter((record) => judgements[record.event_id] === "bad").length;
    counts.textContent = `你的判断：OK ${ok}　有小问题 ${minor}　不行 ${bad}　未判断 ${records.length - ok - minor - bad}`;
  }

  function syncCard(card, eventId) {
    const value = judgements[eventId];
    card.classList.toggle("user-ok", value === "ok");
    card.classList.toggle("user-minor", value === "minor");
    card.classList.toggle("user-bad", value === "bad");
    card.querySelectorAll(".user-review button[data-choice]").forEach((button) => {
      button.classList.toggle("selected", button.dataset.choice === value);
    });
    const note = card.querySelector(".user-note");
    if (note) {
      note.hidden = value !== "minor" && !notes[eventId];
      if (note.value !== (notes[eventId] || "")) note.value = notes[eventId] || "";
    }
  }

  archiveButton.addEventListener("click", () => {
    const items = records
      .filter((record) => judgements[record.event_id] || notes[record.event_id])
      .map((record) => ({
        event_id: record.event_id,
        decision: judgements[record.event_id] || null,
        note: notes[record.event_id] || "",
        original_label: {
          action: record.action,
          tool: record.tool,
          target: record.target,
          left: record.left,
          right: record.right
        }
      }));
    const payload = {
      format: "quiethand-m3-v1-2-user-review",
      version: 1,
      exported_at: new Date().toISOString(),
      event_count: records.length,
      reviewed_count: items.length,
      items
    };
    const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], {type:"application/json"});
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `quiethand-m3-v1-2-review-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  });

  function decorateCard(card) {
    if (card.querySelector(".user-review")) return;
    const eventId = card.querySelector("code")?.textContent;
    const record = recordById.get(eventId);
    if (!record) return;

    const description = naturalDescription(record);
    const judgement = card.querySelector(".judgement");
    judgement.textContent = "";
    const title = document.createElement("div");
    title.className = "model-label-title";
    title.textContent = "模型的中文描述";
    const values = document.createElement("div");
    values.textContent = description;
    judgement.append(title, values);
    card.querySelector(".detail").textContent = missingDescription(record);
    const pill = card.querySelector(".pill");
    pill.textContent = filterNames[record.classification] || "没有结果";
    const rawLink = card.querySelector("a");
    if (rawLink) rawLink.textContent = "查看原视频（不叠标）";

    const states = card.querySelector(".states");
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "查看技术细节（可选）";
    states.parentNode.insertBefore(details, states);
    details.append(summary, states);

    const video = card.querySelector("video");
    const shell = document.createElement("div");
    shell.className = "video-shell";
    const caption = document.createElement("div");
    caption.className = "video-caption";
    caption.textContent = description;
    video.parentNode.insertBefore(shell, video);
    shell.append(video, caption);

    const review = document.createElement("div");
    review.className = "user-review";
    review.innerHTML = '<strong>这条能不能用？</strong><button type="button" data-choice="ok">OK</button><button type="button" data-choice="minor">有小问题</button><button type="button" data-choice="bad">不行</button><button type="button" class="clear">撤销</button><textarea class="user-note" hidden placeholder="例如：刷子实际是黄色，不是绿色；核心动作和双手分工没有问题。"></textarea>';
    review.querySelectorAll("button[data-choice]").forEach((button) => {
      button.addEventListener("click", () => {
        judgements[eventId] = button.dataset.choice;
        save();
        syncCard(card, eventId);
        updateCounts();
      });
    });
    review.querySelector(".user-note").addEventListener("input", (event) => {
      if (event.target.value) notes[eventId] = event.target.value;
      else delete notes[eventId];
      saveNotes();
      syncCard(card, eventId);
    });
    review.querySelector(".clear").addEventListener("click", () => {
      delete judgements[eventId];
      save();
      syncCard(card, eventId);
      updateCounts();
    });
    card.querySelector(".body").appendChild(review);
    syncCard(card, eventId);
  }

  function decorate() {
    document.querySelectorAll(".card").forEach(decorateCard);
  }

  const observer = new MutationObserver(decorate);
  observer.observe(document.querySelector("#grid"), {childList: true});
  decorate();
  updateCounts();
})();
