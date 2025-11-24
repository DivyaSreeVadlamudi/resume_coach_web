import 'dart:js' as js;
import 'dart:js_util' as js_util;
import 'package:flutter/material.dart';

void main() => runApp(const ResumeCoachApp());

class ResumeCoachApp extends StatefulWidget {
  const ResumeCoachApp({super.key});
  @override
  State<ResumeCoachApp> createState() => _ResumeCoachAppState();
}

class _ResumeCoachAppState extends State<ResumeCoachApp> {
  dynamic tfModel;
  dynamic extractor;  // ← This was null before
  bool isLoading = true;
  String result = '';
  final controller = TextEditingController();

  @override
  void initState() {
    super.initState();
    _loadModels();
  }

  Future<void> _loadModels() async {
    try {
      // Wait a bit for TF.js to be ready
      await Future.delayed(const Duration(seconds: 3));

      if (!js.context.hasProperty('tf')) {
        throw Exception('TensorFlow.js not loaded');
      }
      if (!js.context.hasProperty('Xenova')) {
        throw Exception('Transformers.js not loaded');
      }

      // Load your model
      tfModel = await js_util.promiseToFuture(
          js.context.callMethod('tf.loadLayersModel', ['assets/model/model.json'])
      );

      // Load MiniLM (384-dim → we pad to 768)
      extractor = await js_util.promiseToFuture(
          js.context.callMethod('Xenova.pipeline', ['feature-extraction', 'Xenova/all-MiniLM-L6-v2'])
      );

      setState(() {
        isLoading = false;
        result = 'AI Ready! Paste your resume and click "Get Scores"';
      });
    } catch (e) {
      setState(() {
        isLoading = false;
        result = 'Load failed: $e\n\nCheck:\n1. Model files in web/assets/model/\n2. Internet connection\n3. F12 Console';
      });
    }
  }

  Future<void> _analyze() async {
    final text = controller.text.trim();
    if (text.isEmpty || extractor == null || tfModel == null) {
      setState(() => result = 'Paste resume first and wait for "AI Ready"');
      return;
    }

    setState(() => result = 'Analyzing...');

    try {
      final opts = js_util.newObject();
      js_util.setProperty(opts, 'pooling', 'mean');
      js_util.setProperty(opts, 'normalize', true);

      final embedResult = await js_util.promiseToFuture(extractor(text, opts));
      final data = js_util.getProperty(embedResult, 'data') as List<dynamic>;
      final vec384 = data.map((e) => e.toDouble()).toList();

      // Pad 384 → 768
      final vec768 = List<double>.filled(768, 0.0);
      for (int i = 0; i < 384; i++) {
        vec768[i] = vec384[i];
        vec768[i + 384] = vec384[i];
      }

      final input = js.context.callMethod('tf.tensor', [vec768, [1, 768]]);
      final prediction = tfModel.predict(input);
      final raw = await js_util.promiseToFuture(prediction.data());
      input.callMethod('dispose');

      final r = (raw[0] as num).toDouble() * 10;
      final c = (raw[1] as num).toDouble() * 10;
      final p = (raw[2] as num).toDouble() * 10;

      setState(() {
        result = '''
YOUR RESUME SCORES (0–100)

Readability:     ${r.toStringAsFixed(1)}
Clarity:         ${c.toStringAsFixed(1)}
Professionalism: ${p.toStringAsFixed(1)}
────────────────────
AVERAGE:         ${((r + c + p) / 3).toStringAsFixed(1)}/100
        '''.trim();
      });
    } catch (e) {
      setState(() => result = 'Analysis failed: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        appBar: AppBar(
          title: const Text('Resume Coach AI'),
          backgroundColor: Colors.deepPurple,
          foregroundColor: Colors.white,
        ),
        body: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            children: [
              TextField(
                controller: controller,
                maxLines: 10,
                decoration: const InputDecoration(
                  border: OutlineInputBorder(),
                  hintText: 'Paste your full resume here...',
                ),
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: isLoading ? null : _analyze,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green[700],
                  padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 16),
                ),
                child: Text(
                  isLoading ? 'Loading AI...' : 'GET SCORES (0–100)',
                  style: const TextStyle(fontSize: 18, color: Colors.white),
                ),
              ),
              const SizedBox(height: 30),
              SelectableText(
                result,
                style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      ),
    );
  }
}