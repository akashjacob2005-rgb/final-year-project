const API = "http://127.0.0.1:8000/api";

let currentTask = 0;

let scores = {

    memory: 50,

    attention: 50,

    language: 50,

    executive: 50,

    reaction: 50

};


const tasks = [

    {
        title: "Memory Recall",
        description:
            "Remember the five items. You will be asked to recall them later.",
        render: memoryTask
    },

    {
        title: "Processing Speed",
        description:
            "Measure your visual reaction time.",
        render: reactionTask
    },

    {
        title: "Attention",
        description:
            "Identify the number of target items in the sequence.",
        render: attentionTask
    },

    {
        title: "Working Memory",
        description:
            "Remember and reproduce the sequence in the same order.",
        render: workingMemoryTask
    },

    {
        title: "Language Fluency",
        description:
            "Generate as many words as possible beginning with F.",
        render: languageTask
    },

    {
        title: "Executive Function",
        description:
            "Arrange the numbers from smallest to largest.",
        render: executiveTask
    }

];


/* ================= NAVIGATION ================= */

function showPage(pageId, button = null) {

    document
        .querySelectorAll(".page")
        .forEach(function(page) {

            page.classList.remove("active");

        });


    const page =
        document.getElementById(pageId);


    if (page) {

        page.classList.add("active");

    }


    document
        .querySelectorAll(".nav-item")
        .forEach(function(item) {

            item.classList.remove("active");

        });


    if (button) {

        button.classList.add("active");

    }


    const titles = {

        dashboard:
            "Cognitive Health Overview",

        assessment:
            "Interactive Cognitive Assessment",

        results:
            "Cognitive Analysis",

        monitoring:
            "Longitudinal Monitoring",

        uploads:
            "Data & Reports"

    };


    const breadcrumbs = {

        dashboard:
            "DASHBOARD",

        assessment:
            "ASSESSMENT",

        results:
            "AI ANALYSIS",

        monitoring:
            "MONITORING",

        uploads:
            "DATA"

    };


    document.getElementById(
        "pageTitle"
    ).textContent =
        titles[pageId];


    document.getElementById(
        "breadcrumb"
    ).textContent =
        breadcrumbs[pageId];


    if (pageId === "assessment") {

        renderTask();

    }

}


/* ================= ASSESSMENT ================= */

function renderTask() {

    const task =
        tasks[currentTask];


    document.getElementById(
        "taskTitle"
    ).textContent =
        task.title;


    document.getElementById(
        "taskDescription"
    ).textContent =
        task.description;


    document.getElementById(
        "taskCounter"
    ).textContent =
        `${currentTask + 1} / ${tasks.length}`;


    document.getElementById(
        "progressBar"
    ).style.width =
        `${((currentTask + 1) / tasks.length) * 100}%`;


    for (
        let i = 0;
        i < tasks.length;
        i++
    ) {

        const menu =
            document.getElementById(
                `taskMenu${i}`
            );


        if (menu) {

            menu.classList.remove(
                "active"
            );

        }

    }


    document
        .getElementById(
            `taskMenu${currentTask}`
        )
        .classList.add(
            "active"
        );


    document.getElementById(
        "taskContent"
    ).innerHTML = "";


    task.render();


    document.getElementById(
        "backBtn"
    ).style.visibility =
        currentTask === 0
            ? "hidden"
            : "visible";


    document.getElementById(
        "nextBtn"
    ).textContent =
        currentTask === tasks.length - 1
            ? "Finish Assessment →"
            : "Continue →";

}


/* ================= MEMORY ================= */

function memoryTask() {

    document.getElementById(
        "taskContent"
    ).innerHTML = `

        <div class="task-inner">

            <span class="eyebrow">
                MEMORY ENCODING
            </span>

            <h3>
                Remember these items
            </h3>

            <p>
                Study the five objects below.
                Try to remember them as accurately as possible.
            </p>

            <div class="memory-items">

                <div class="memory-item">
                    APPLE
                </div>

                <div class="memory-item">
                    RIVER
                </div>

                <div class="memory-item">
                    CHAIR
                </div>

                <div class="memory-item">
                    BOOK
                </div>

                <div class="memory-item">
                    FLOWER
                </div>

            </div>

            <p>
                You will be asked to recall them
                during a later stage of the assessment.
            </p>

        </div>

    `;

}


/* ================= REACTION ================= */

function reactionTask() {

    document.getElementById(
        "taskContent"
    ).innerHTML = `

        <div class="task-inner">

            <span class="eyebrow">
                PROCESSING SPEED
            </span>

            <h3>
                Reaction Time
            </h3>

            <p>
                Wait until the target changes.
                Click as quickly as possible.
            </p>

            <div
                id="reactionCircle"
                onclick="reactionClicked()"
                style="
                    width:150px;
                    height:150px;
                    margin:30px auto;
                    border-radius:50%;
                    background:#132b25;
                    border:1px solid #28584b;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    cursor:pointer;
                    color:#7e9890;
                    font-size:11px;
                    font-weight:800;
                "
            >

                WAIT

            </div>

            <p id="reactionResult">
                Preparing test...
            </p>

        </div>

    `;


    window.reactionStart = null;


    setTimeout(function() {

        const circle =
            document.getElementById(
                "reactionCircle"
            );


        if (!circle) {

            return;

        }


        circle.style.background =
            "#1b806a";

        circle.style.color =
            "#ffffff";

        circle.style.borderColor =
            "#35d6ad";

        circle.textContent =
            "CLICK";


        window.reactionStart =
            performance.now();

    }, 1700);

}


function reactionClicked() {

    if (!window.reactionStart) {

        return;

    }


    const time =
        Math.round(
            performance.now()
            -
            window.reactionStart
        );


    scores.reaction =
        Math.max(
            0,
            Math.min(
                100,
                100 -
                ((time - 200) / 10)
            )
        );


    document.getElementById(
        "reactionResult"
    ).textContent =
        `Reaction time recorded: ${time} ms`;

}


/* ================= ATTENTION ================= */

function attentionTask() {

    document.getElementById(
        "taskContent"
    ).innerHTML = `

        <div class="task-inner">

            <span class="eyebrow">
                SELECTIVE ATTENTION
            </span>

            <h3>
                Count the target numbers
            </h3>

            <p>
                How many times does the number
                <strong>7</strong> appear?
            </p>

            <div
                class="memory-items"
                style="
                    font-size:22px;
                    letter-spacing:7px;
                "
            >

                3 7 2 9 7 1 4 7
                8 2 6 7 5 1 9 7

            </div>

            <input
                id="attentionAnswer"
                class="big-input"
                type="number"
                placeholder="Enter your answer"
            >

        </div>

    `;

}


/* ================= WORKING MEMORY ================= */

function workingMemoryTask() {

    document.getElementById(
        "taskContent"
    ).innerHTML = `

        <div class="task-inner">

            <span class="eyebrow">
                WORKING MEMORY
            </span>

            <h3>
                Remember the sequence
            </h3>

            <p>
                Reproduce the sequence exactly.
            </p>

            <div
                class="memory-items"
                style="
                    font-size:25px;
                "
            >

                <div class="memory-item">8</div>

                <div class="memory-item">3</div>

                <div class="memory-item">1</div>

                <div class="memory-item">9</div>

                <div class="memory-item">4</div>

            </div>

            <input
                id="workingAnswer"
                class="big-input"
                placeholder="Example: 8 3 1 9 4"
            >

        </div>

    `;

}


/* ================= LANGUAGE ================= */

function languageTask() {

    document.getElementById(
        "taskContent"
    ).innerHTML = `

        <div class="task-inner">

            <span class="eyebrow">
                VERBAL FLUENCY
            </span>

            <h3>
                Generate words
            </h3>

            <p>
                Enter as many different words beginning
                with <strong>F</strong> as you can.
            </p>

            <textarea
                id="languageAnswer"
                class="big-input"
                style="height:150px; resize:none;"
                placeholder="
forest
family
flower
future
...
"
            ></textarea>

        </div>

    `;

}


/* ================= EXECUTIVE ================= */

function executiveTask() {

    document.getElementById(
        "taskContent"
    ).innerHTML = `

        <div class="task-inner">

            <span class="eyebrow">
                EXECUTIVE FUNCTION
            </span>

            <h3>
                Sequence the numbers
            </h3>

            <p>
                Arrange the following numbers
                from smallest to largest.
            </p>

            <div
                class="memory-items"
                style="
                    font-size:20px;
                "
            >

                <div class="memory-item">8</div>
                <div class="memory-item">3</div>
                <div class="memory-item">9</div>
                <div class="memory-item">1</div>
                <div class="memory-item">6</div>
                <div class="memory-item">2</div>
                <div class="memory-item">5</div>

            </div>

            <input
                id="executiveAnswer"
                class="big-input"
                placeholder="Example: 1 2 3 5 6 8 9"
            >

        </div>

    `;

}


/* ================= SAVE TASK ================= */

function saveCurrentTask() {

    if (currentTask === 0) {

        scores.memory = 80;

    }


    if (currentTask === 1) {

        if (!window.reactionStart) {

            scores.reaction = 50;

        }

    }


    if (currentTask === 2) {

        const value =
            Number(
                document.getElementById(
                    "attentionAnswer"
                )?.value
            );


        if (value === 5) {

            scores.attention = 100;

        }

        else {

            scores.attention =
                Math.max(
                    0,
                    100 -
                    Math.abs(
                        5 - value
                    ) * 20
                );

        }

    }


    if (currentTask === 3) {

        const value =
            document.getElementById(
                "workingAnswer"
            )?.value
            .trim();


        scores.memory =
            value === "8 3 1 9 4"
                ? 100
                : 40;

    }


    if (currentTask === 4) {

        const text =
            document.getElementById(
                "languageAnswer"
            )?.value || "";


        const words =
            text
            .split(/[\s,]+/)
            .filter(Boolean);


        const unique =
            new Set(
                words.map(
                    word =>
                        word.toLowerCase()
                )
            );


        scores.language =
            Math.min(
                100,
                unique.size * 10
            );

    }


    if (currentTask === 5) {

        const value =
            document.getElementById(
                "executiveAnswer"
            )?.value
            .trim();


        scores.executive =
            value ===
            "1 2 3 5 6 8 9"
                ? 100
                : 40;

    }

}


/* ================= NEXT ================= */

function nextTask() {

    saveCurrentTask();


    if (
        currentTask <
        tasks.length - 1
    ) {

        currentTask++;

        renderTask();

    }

    else {

        submitAssessment();

    }

}


/* ================= BACK ================= */

function previousTask() {

    saveCurrentTask();


    if (currentTask > 0) {

        currentTask--;

        renderTask();

    }

}


/* ================= SUBMIT ================= */

async function submitAssessment() {

    showPage("results");


    document.getElementById(
        "riskNumber"
    ).textContent =
        "…";


    try {

        const payload = {

            memory_score:
                scores.memory,

            attention_score:
                scores.attention,

            language_score:
                scores.language,

            executive_score:
                scores.executive,

            reaction_time_ms:
                1200 -
                (
                    scores.reaction
                    -
                    50
                ) * 10,

            age: 60

        };


        const response =
            await fetch(
                API + "/assess",
                {

                    method:
                        "POST",

                    headers: {

                        "Content-Type":
                            "application/json"

                    },

                    body:
                        JSON.stringify(
                            payload
                        )

                }
            );


        const data =
            await response.json();


        displayResults(data);

    }

    catch (error) {

        console.log(
            "Backend unavailable. Using demo calculation."
        );


        const average =
            (
                scores.memory +
                scores.attention +
                scores.language +
                scores.executive +
                scores.reaction
            )
            / 5;


        const risk =
            Math.max(
                0,
                Math.round(
                    100 - average
                )
            );


        displayResults({

            risk_percent_demo:
                risk,

            risk_level_demo:

                risk < 25
                    ? "Low"
                    : risk < 55
                        ? "Moderate"
                        : "Elevated",

            component_scores: {

                memory:
                    scores.memory,

                attention:
                    scores.attention,

                language:
                    scores.language,

                executive:
                    scores.executive,

                processing_speed:
                    scores.reaction

            }

        });

    }

}


/* ================= DISPLAY RESULTS ================= */

function displayResults(data) {

    const risk =
        data.risk_percent_demo ?? 0;


    document.getElementById(
        "riskNumber"
    ).textContent =
        risk + "%";


    document.getElementById(
        "riskLabel"
    ).textContent =
        data.risk_level_demo ||
        "Prototype indicator";


    document.getElementById(
        "summaryText"
    ).textContent =

        "NeuroGuard integrates performance across memory, attention, language, executive function and processing speed. In the complete research system, these behavioral features will be combined with NLP, CNN and graph-learning representations trained using appropriate research datasets.";


    const components =
        data.component_scores ||
        scores;


    const domains = [

        [
            "Memory",
            components.memory
        ],

        [
            "Attention",
            components.attention
        ],

        [
            "Language",
            components.language
        ],

        [
            "Executive Function",
            components.executive
        ],

        [
            "Processing Speed",
            components.processing_speed ||
            components.reaction ||
            scores.reaction
        ]

    ];


    const grid =
        document.getElementById(
            "domainGrid"
        );


    grid.innerHTML = "";


    domains.forEach(
        function(domain) {

            const value =
                Math.round(
                    domain[1] || 0
                );


            const card =
                document.createElement(
                    "div"
                );


            card.className =
                "result-domain-card";


            card.innerHTML = `

                <small>
                    ${domain[0]}
                </small>

                <strong>
                    ${value}
                </strong>

                <div class="result-domain-bar">

                    <div
                        style="
                            width:${Math.min(
                                100,
                                Math.max(
                                    0,
                                    value
                                )
                            )}%;
                        "
                    ></div>

                </div>

            `;


            grid.appendChild(card);

        }
    );


    document.getElementById(
        "dashboardRisk"
    ).textContent =
        risk + "%";


    document.getElementById(
        "assessmentCount"
    ).textContent =
        "1";


    updateRiskCircle(risk);

}


/* ================= RISK CIRCLE ================= */

function updateRiskCircle(risk) {

    const circle =
        document.querySelector(
            ".risk-circle"
        );


    if (!circle) {

        return;

    }


    const degrees =
        Math.min(
            100,
            Math.max(
                0,
                risk
            )
        )
        * 3.6;


    circle.style.background = `

        radial-gradient(
            circle,
            #0a1916 58%,
            transparent 59%
        ),

        conic-gradient(
            #35d6ad ${degrees}deg,
            #1d3d35 ${degrees}deg
        )

    `;

}


/* ================= FILE UPLOAD ================= */

function fileSelected(input) {

    if (
        !input.files ||
        input.files.length === 0
    ) {

        return;

    }


    const file =
        input.files[0];


    const parent =
        input.parentElement;


    const text =
        parent.querySelector(
            "strong"
        );


    if (text) {

        text.textContent =
            file.name;

    }

}