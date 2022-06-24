<?php

$html = '';
if (empty($_GET['im']) === false) {
    $imagePath = 'avengers/' . $_GET['im'] . '.jpg';
    if (file_exists($imagePath)) {
        if (empty($_GET['show']) === false) {
            $html = '<img src="' . $imagePath . '" width="90%">';
        } else {
            header($_SERVER["SERVER_PROTOCOL"] . " 200 OK");
            header("Content-Type: application/octet-stream"); //zip
            header("Content-Transfer-Encoding: Binary");
            header("Content-Length:".filesize($imagePath));
            header("Content-Disposition: attachment; filename=" . $imagePath);
            readfile($imagePath);
            die();
        }        
    }
}

?>

<!DOCTYPE html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
        <meta name="title" content="Selfie with...">
        <meta name="description" content="Take selfie with the stars">
        <title>Selfie with...</title>
        <style>
            body {
                background: #000;
                margin: 0px;
                font-family: Helvetica, sans-serif, Verdana;
            }
            .content {
                width: 100vw;
                height: 100vh;
                margin: auto;
                overflow-y: auto;
                overflow-x: hidden;
                display: flex;
                align-items: center;
                justify-content: center;
                position: fixed;
            }
            .img {
                height: auto;
                margin: auto;
                display: flex;
                align-items: center;
                justify-content: center;
                position: fixed;
            }
        </style>
    </head>
    <body>
        <div class="content">
            <div class="img"><?php echo $html; ?></div>
        </div>
    </body>
</html>