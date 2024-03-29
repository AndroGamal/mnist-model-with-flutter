import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:tflite/tflite.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const MyHomePage(title: 'Flutter Demo Home Page'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key, required this.title}) : super(key: key);
  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  void convolution() {
    () async {
      try {
        FilePickerResult? result = await FilePicker.platform.pickFiles();
        if (result != null) {
          await Tflite.loadModel(
              model: "assets/convolution.tflite", labels: "assets/labels.txt");
          var output = await Tflite.runModelOnImage(
              path: result.files.single.path ?? "", numResults: 10);
          setState(() {
            text = output![0]['label'].toString();
          });
          await Tflite.close();
        } else {}
      } catch (ex) {
        setState(() {
          text = ex.toString();
        });
      }
    }.call();
  }

  void HmorHs() {
    () async {
      try {
        FilePickerResult? result = await FilePicker.platform.pickFiles();
        if (result != null) {
          await Tflite.loadModel(
              model: "assets/human_or_horse.tflite",
              labels: "assets/human_or_horse.txt");
          var output = await Tflite.runModelOnImage(
              path: result.files.single.path ?? "", numResults: 4);
          setState(() {
            text = output![0]['label'].toString();
          });
          await Tflite.close();
        } else {}
      } catch (ex) {
        setState(() {
          text = ex.toString();
        });
      }
    }.call();
  }

  void model() {
    () async {
      try {
        var interpreter = await tfl.Interpreter.fromAsset("model.tflite");
        var input = [20.0];
        var output = List.filled(1 * 1, 0).reshape([1, 1]);
        interpreter.run(input, output);
        interpreter.close();
        setState(() {
          text = output.toString();
        });
      } catch (ex) {
        setState(() {
          text = ex.toString();
        });
      }
    }.call();
  }

  dynamic method;
  String text = "test";
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            RadioListTile(
                title: Text("prediction numder"),
                value: model,
                groupValue: method,
                onChanged: (va) {
                  setState(() {
                    method = va;
                  });
                }),
            RadioListTile(
                title: Text("mnist model"),
                value: convolution,
                groupValue: method,
                onChanged: (va) {
                  setState(() {
                    method = va;
                  });
                }),
            RadioListTile(
                title: Text("humman or hourse"),
                value: HmorHs,
                groupValue: method,
                onChanged: (va) {
                  setState(() {
                    method = va;
                  });
                }),
            Text(text),
            FloatingActionButton(
                child: Icon(Icons.add_a_photo_rounded),
                onPressed: () {
                  method();
                })
          ],
        ),
      ),
    );
  }
}
