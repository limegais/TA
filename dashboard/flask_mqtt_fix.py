# FLASK APP.PY FIX - Replace learn_ir() function with this:

@app.route('/api/ir/learn', methods=['POST'])
def learn_ir():
    global ir_learning_mode, ir_learning_button, ir_learning_device
    try:
        data = request.json
        button_name = data.get('button', '')
        device_name = data.get('device', 'remote')
        
        if not button_name:
            return jsonify({'status': 'error', 'message': 'Button name required'}), 400
        
        ir_learning_mode = True
        ir_learning_button = button_name
        ir_learning_device = device_name
        
        # MQTT PAYLOAD
        mqtt_payload = {
            'button': button_name, 
            'device': device_name,
            'action': 'start'
        }
        mqtt_payload_str = json.dumps(mqtt_payload)
        
        # DEBUG LOGGING - SEE WHAT WE'RE SENDING!
        print("\n" + "="*60)
        print("🔴 FLASK: PUBLISHING IR LEARN COMMAND")
        print("="*60)
        print(f"Topic    : smartroom/ir/learn")
        print(f"Button   : {button_name}")
        print(f"Device   : {device_name}")
        print(f"Payload  : {mqtt_payload_str}")
        print(f"Length   : {len(mqtt_payload_str)} bytes")
        print(f"MQTT Connected: {mqtt_client.is_connected()}")
        print("="*60)
        
        # PUBLISH WITH RESULT CHECK
        result = mqtt_client.publish('smartroom/ir/learn', mqtt_payload_str)
        
        print(f"Publish Result: {result.rc}")
        if result.rc == 0:
            print("✅ MQTT PUBLISH SUCCESS!")
        else:
            print(f"❌ MQTT PUBLISH FAILED! RC={result.rc}")
        print("="*60 + "\n")
        
        log_messages.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'msg': f'IR Learning started for: {device_name} - {button_name}',
            'level': 'info'
        })
        
        return jsonify({
            'status': 'success', 
            'message': f'Learning mode activated for {device_name} - {button_name}',
            'mqtt_published': result.rc == 0
        })
    except Exception as e:
        print(f"❌ EXCEPTION in learn_ir(): {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500
