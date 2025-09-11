import 'package:flutter/material.dart';

class Bbox extends StatelessWidget {
  final double x;
  final double y;
  final double width;
  final double height;
  final String label;
  final double score;
  final Color color;

  const Bbox(
    this.x,
    this.y,
    this.width,
    this.height,
    this.label,
    this.score,
    this.color, {
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: y - height / 2,
      left: x - width / 2,
      width: width,
      height: height,
      child: Container(
        decoration: BoxDecoration(
          border: Border.all(color: color, width: 3),
          borderRadius: const BorderRadius.all(Radius.circular(4)),
        ),
        child: label.isEmpty
            ? const SizedBox.shrink()
            : Align(
                alignment: Alignment.topLeft,
                child: FittedBox(
                  child: Container(
                    color: color,
                    padding: const EdgeInsets.symmetric(horizontal: 2, vertical: 1),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: <Widget>[
                        Text(label, style: const TextStyle(color: Colors.white)),
                        Text(' ${(score * 100).toStringAsFixed(0)}%', style: const TextStyle(color: Colors.white)),
                      ],
                    ),
                  ),
                ),
              ),
      ),
    );
  }
}
