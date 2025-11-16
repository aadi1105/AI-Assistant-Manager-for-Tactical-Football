let reportData = [];
document.addEventListener("DOMContentLoaded", async () => {
    const params = new URLSearchParams(window.location.search);

    const home = params.get("home");
    const away = params.get("away");

    if (!home || !away) {
        console.error("Missing home or away parameters in URL.");
        return;
    }

    try {
        const res = await fetch(`http://127.0.0.1:5000/pregenerate?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}`);
        const data = await res.json();

        // store the response in an array
        reportData = Array.isArray(data) ? data : [];

        console.log("Pregenerate response:", reportData);

        let titleOne = document.getElementById('title-0');
        let titleTwo = document.getElementById('title-1');
        let titleThree = document.getElementById('title-2');
        let titleFour = document.getElementById('title-3');
        let titleFive = document.getElementById('title-4');
        let titleSix = document.getElementById('title-5');
        let titleSeven = document.getElementById('title-6');
        
        titleOne.innerText = reportData[0];
        titleTwo.innerText = reportData[1];
        titleThree.innerText = reportData[2];
        titleFour.innerText = reportData[3];
        titleFive.innerText = reportData[4];
        titleSix.innerText = reportData[5];
        titleSeven.innerText = reportData[6];

    } catch (err) {
        console.error("Error fetching pregenerate:", err);
    }
});
